# #!/bin/bash
echo "Downloading a product"
python -m utils.downloader.search --start "2021-01-01" --end "2023-12-31" --out "./DATA/" --download --max 20
python -m utils.downloader.unzipper 
