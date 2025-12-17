# COCO Dataset Setup

We will use the **COCO 2017 Validation Set** (approx. 5000 images, ~1GB) as our training data.
This is smaller than the full Training Set (18GB) but sufficient for this project.

## Step 1: Download
Download the "2017 Val images" from the official COCO website:
**Link**: [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)

## Step 2: Extract
Extract the downloaded `val2017.zip` file into the `src/data/` folder.

## Expected Structure
After extraction, your folder structure should look like this:

```
src/
  data/
    val2017/        <-- Folder containing .jpg images
      000000000139.jpg
      ...
  style_transfer/
  main.py
  ...
```

**Note**: You do NOT need to download the annotations for this project, just the images.
