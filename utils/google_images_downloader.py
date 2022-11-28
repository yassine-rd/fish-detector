# python script to download image from google.com 
# do not install the library using the `pip install google_images_download` command
# instead, install it using : pip install git+https://github.com/Joeclinton1/google-images-download.git

# importing the library
from google_images_download import google_images_download

# class instantiation
response = google_images_download.googleimagesdownload()

# creating list of arguments
# keywords are 'paracheirodon innesi' for neon_bleu class and 'paracheirodon axelrodi' for cardinalis class
arguments = {"keywords": "paracheirodon axelrodi",
             "limit": 100,
             "format": "jpg",
             "image_directory" : "paracheirodon-axelrodi",
             "print_urls": False}

# passing the arguments to the function
paths = response.download(arguments)

# printing absolute paths of the downloaded images
# print(paths)
