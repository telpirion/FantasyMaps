# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.cloud import firestore
from google.cloud import storage

from PIL import Image    

import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import praw
import re
import requests
import shutil
import spacy

def get_reddit_posts(reddit_credentials, subreddit_name, limit):
    """Gets the top (hot) posts from a subreddit.
    
    Arguments:
        reddit_credentials (dict): a dictionary with client_id, secret, and user_agent
        subreddit_name (str): the name of the subreddit to scrape posts from
        limit (int): the maximum number of posts to grab
    
    Returns:
        List of Reddit API objects
    """

    reddit = praw.Reddit(client_id=reddit_credentials["client_id"], 
                 client_secret=reddit_credentials["secret"],
                 user_agent=reddit_credentials["user_agent"])
    
    return reddit.subreddit(subreddit_name).hot(limit=limit)


def convert_posts_to_dataframe(posts, columns):
    """Convert a list of Reddit posts into a pandas.Dataframe.
    
    Arguments:
        posts (list): a list of Reddit posts
        columns: the columns to use for the Dataframe
    
    Returns:
        a pandas.Dataframe
    """
    
    filtered_posts = [[s.title, s.selftext, s.id, s.url] for s in posts]
    filtered_posts = np.array(filtered_posts)
    reddit_posts_df = pd.DataFrame(filtered_posts,
                               columns=columns)

    return reddit_posts_df

def make_nice_filename(name):
    """Converts Reddit post title into a meaningful(ish) filename.
    
    Arguments:
        name (str): title of the post
    
    Returns:
        String. Format is `<adj.>-<nouns>.<cols>x<rows>.jpg`
    """
    
    dims = re.findall("\d+x\d+", name)
    if len(dims) is 0:
        return ""
    
    dims = dims[0].split("x")
    if len(dims) is not 2:
        return ""
    
    tokens = get_tokens(name)
    new_name = name.lower()[:30]
    
    if len(tokens) > 0:
        tokens = tokens[:6] # Arbitrarily keep new names to six words or less
        new_name = "_".join(tokens)
    
    return f"{new_name}.{dims[0]}x{dims[1]}.jpg"

def get_tokens(title):
    """Analyzes a post for nouns, proper nouns, and adjectives.
    
    Arguments:
        title (str): title of the post
    
    Returns:
        List of string. Words to use in a filename.    
    """
    POS = ["PROPN", "NOUN", "ADJ"]
    
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    
    words = []
    
    tokens = nlp(title)
    for t in tokens:
        pos = t.pos_
        
        if pos in POS:
            words.append(t.text.lower())
    
    return words 

def convert_image_to_hash(content, hashes):
    """Convert image data to hash value (str).
    
    Arguments:
        content (byte array): the image
        hashes (list): a list of hashes from converted strings
    
    Return:
        Bool. Indicates whether the process was success.
    """

    sha1 = hashlib.sha1()
    jpg_hash = sha1.update(content)
    jpg_hash = sha1.hexdigest()
        
    if jpg_hash in hashes:
        hashes.append("")
        return False

    hashes.append(jpg_hash)
    return True

def download_image_local(url, path, hashes):
    """Download an image from the internet to local file system.
    
    Arguments:
        url (str): the image to download
        path (str): the local path to save the image.
        hashes (list): the list of UIDs for downloaded images
    
    Returns:
        Bool. Indicates whether downloading the image was successful.
    """
    
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        
        is_unique = convert_image_to_hash(r.content, hashes)
        if not is_unique:
            return False
        
        with open(path, 'wb') as f:
            f.write(r.content)
    else:
        return False
    
    return True

def get_image_width_and_height(path):
    """Open the image and get the image's height and width in pixels.
    
    Arguments:
        path (str):
        
    Returns:
        Tuple of width, height
    """
    
    img = Image.open(path)
    w, h = img.size
    
    return (math.floor(w), math.floor(h))

def compute_vtt_data(width, height, columns, rows):
    """Calculate the VTT values for the image.
    
    Arguments:
        width (int):
        height (int):
        columns (int):
        rows (int): 
    Returns:
        Dict.
    """
    
    return {
        "cellsOffsetX": 0, # Assumes no offset
        "cellsOffsetY": 0, # Assumes no offset
        "imageWidth": int(width),
        "imageHeight": int(height),
        "cellWidth": int(width / columns),
        "cellHeight": int(height / rows)
    }

def compute_shard_coordinates(width, height, cell_width,
                              cell_height, columns, rows):
    """Converts image data into 1,or more shards.
    
    Arguments:
        width (int):
        height (int):
        cell_width (int):
        cell_height (int):
        columns (int):
        rows (int):
        
    Returns:
        List of tuples of (xMin, yMin, xMax, yMax, columns, rows)
        TODO: convert to pd.Series
    """
    
    total_cells = columns * rows
    if total_cells <= 500:
        return
    
    # Assume that a perfectly square map that approaches 500 cells is 22 cols by 22 rows.
    # Cut an image into as many 22x22 shards as possible
    SQRT = 22
    
    h_shards = math.floor(columns / SQRT)
    h_rem = columns % SQRT
    v_shards = math.floor(rows / SQRT)
    v_rem = rows % SQRT
    shard_columns = shard_rows = SQRT
    
    # Edge case 1: we have a narrow width (portrait-oriented) map
    if h_shards == 0:
        h_shards = 1
        h_rem = 0
        shard_columns = columns
    
    # Edge case 2: we have a short height (landscape-oriented) map
    if v_shards == 0:
        v_shards = 1
        v_rem = 0
        shard_rows = rows
    
    shards = []
    curr_min_x = 0
    curr_min_y = 0
    for _ in range(h_shards):
        max_x = (cell_width * shard_columns) + curr_min_x
        if max_x > width:
            max_x = width
        for _ in range(v_shards):
            max_y = (cell_height * shard_rows) + curr_min_y
            if max_y > height:
                max_y = height
            
            shards.append((curr_min_x, curr_min_y, max_x, max_y, shard_columns, shard_rows))
            curr_min_y = max_y
            
        curr_min_y = 0
        curr_min_x = max_x
    
    # Get the right-side remainder
    curr_min_x = width - (h_rem * cell_width)
    curr_min_y = 0
    for _ in range(v_shards):
        max_y = (cell_height * shard_rows) + curr_min_y
        if max_y > height:
            max_y = height
        shards.append((curr_min_x, curr_min_y, width, max_y, h_rem, shard_rows))
        curr_min_y = max_y
    
    # Get the bottom-side remainder
    curr_min_y = height - (v_rem * cell_height)
    curr_min_x = 0
    for _ in range(h_shards):
        max_x = (cell_width * shard_columns) + curr_min_x
        if max_x > width:
            max_x = width
        shards.append((curr_min_x, curr_min_y, max_x, height, shard_columns, v_rem))
        curr_min_x = max_x
            
    return shards

def create_shard(x_min, y_min, x_max, y_max, cols, rows, img_path, parent_id):
    """Crops and saves an image.
    
    Arguments:
        x_min (int): the left-most point to crop, relative to the parent image
        y_min (int): the top-most point to crop, relative to the parent image
        x_max (int): the right-most point, relative to the parent image
        y_max (int): the bottom-most poinst, relative to the parent image
        cols (cols): the grid columns in this shard
        rows (rows): the grid rows in this shard
        img_path (str): the parent image's local path
        parent_id (str): the parent image's UID
    
    Returns:
        DataFrame with local path, UID, width, height, columns, and rows
    
    """
    try:

        img = Image.open(img_path)
        shard = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Get new filepath name
        s_path = create_shard_path(img_path, x_min, y_min, cols, rows)

        # Get new UID
        hashes = []
        convert_image_to_hash(shard.tobytes(), hashes)

        shard.save(s_path)
        
        d = {
            "Width": math.floor(x_max - x_min),
            "Height": math.floor(y_max - y_min),
            "Columns": cols,
            "Rows": rows,
            "UID": hashes[0],
            "Path": s_path,
            "IsShard": True,
            "Parent": parent_id
        }
        
    except SystemError as e:
        print(f"Error: {img_path}, bounds: {x_max},{y_max}")
        return None
    
    return pd.DataFrame(data=d, index=[0])

def create_shard_path(path, x_min, y_min, cols, rows):
    """Convert an image path string to new string.
    
    Assumes the image path is of the format:
        <folder>/<name>.<cols>x<rows>.jpg
    
    Arguments:
        path (str):
        x_min (int):
        y_min (int):
        cols (int):
        rows (int):
        
    Returns:
        String. New image path.
    """
    
    paths = path.split(".")
    paths[-2] = f"{math.floor(x_min)}_{math.floor(y_min)}.{cols}x{rows}"
    s_path = ".".join(paths)
    return s_path

def compute_bboxes(*, dataframe=None, series=None, cell_width=0, cell_height=0):
    """Determines bounding boxes for image object detection.
    
    Arguments:
        dataframe (pandas.Dataframe): A DataFrame with Height, Width, Columns, and Rows
        series (pandas.Series): A Series with Height, Width, Columns, and Rows
        cell_width (int):
        cell_height (int):
    
    Returns:
        List of dict.
    """
    bboxes = []
    try:
        if dataframe is not None:
            width = dataframe.iloc[0]["Width"]
            height = dataframe.iloc[0]["Height"]
            columns = dataframe.iloc[0]["Columns"]
            rows = dataframe.iloc[0]["Rows"]
        elif series is not None:
            width = series["Width"]
            height = series["Height"]
            columns = series["Columns"]
            rows = series["Rows"]
        else:
            return bboxes

        BORDER = 1 # 1px border around the outside of the cell
        LABEL = "cell"

        curr_x = cell_width
        while curr_x < width:
            curr_y = cell_height
            while curr_y < height:
                x_min = (curr_x - BORDER) / width
                y_min = (curr_y - BORDER) / height
                x_max = (curr_x + cell_width + BORDER) / width
                y_max = (curr_y + cell_height + BORDER) / height
                bboxes.append({
                    "xMin": x_min,
                    "xMax": x_max,
                    "yMin": y_min,
                    "yMax": y_max,
                    "displayName": LABEL
                })
                curr_y = curr_y + cell_height
            curr_x = curr_x + cell_width
    except:
        print(f"Error: {dataframe}")
        
    return bboxes

def store_image_gcs(*, project_id, series, bucket_name, prefix):
    """Copies a local image to Google Cloud Storage.
    
    Arguments:
        project_id (str): the Google Cloud Project ID to use
        series (pd.Series): a Pandas Series with "Path" column
        bucket_name (str): the Cloud Storage bucket to use
        prefix (str): the prefix or "folder" to use in the bucket

    Returns:
        String. The Cloud Storage URI of the image.
    """
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    local_path = series["Path"]
    file_name = local_path.split("/")[-1]
    img_gcs_uri = f"gs://{bucket_name}/{prefix}/{file_name}"
    blob_name = f"{prefix}/{file_name}"
            
    file_blob = bucket.blob(blob_name)           
    file_blob.upload_from_filename(local_path)
    
    return img_gcs_uri

def store_metadata_fs(*, project_id, series, collection_name, uid):
    """Upserts image metadata into a Firestore collection.
    
    Arguments:
        project_id (str): the Google Cloud project to store these in
        series (pd.Series): a Pandas series with the image's metadata
        collection_name (str): the Firestore collection to store the data in
    """
    
    client = firestore.Client(project=project_id)
    
    series_dict = series.to_dict()
    
    # clean up the data a little bit before upserting
    vtt = series["VTT"]
    if vtt is not "":
        vtt = json.loads(vtt)
        series_dict["VTT"] = vtt
        
    bboxes = series["BBoxes"]
    if bboxes is not "":
        bboxes = json.loads(bboxes)["bboxes"]
        series_dict["BBoxes"] = bboxes
    
    file_name = series["Path"].split("/")[-1]
    series_dict.pop("Path", None)
    series_dict["filename"] = file_name
    
    img_gcs_uri = series["GCS URI"]
    series_dict.pop("GCS URI", None)
    series_dict["gcsURI"] = img_gcs_uri
    
    # upsert the dict directly into Firestore!
    client.collection(collection_name).document(uid).set(series_dict)