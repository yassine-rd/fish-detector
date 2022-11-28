# python script to download image from bing.com 
# installation : pip install bing-image-downloader

# importing the library
from bing_image_downloader import downloader

# downloading bing images
# 'paracheirodon innesi' for neon_bleu class and 'paracheirodon axelrodi' for cardinalis class
downloader.download("paracheirodon axelrodi",
                    limit =200, 
                    output_dir='./paracheirodon-axelrodi')