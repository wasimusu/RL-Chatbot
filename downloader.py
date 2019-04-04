import urllib.request

bigdataset_aurl = "http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-eu.txt.zip"
print("Downloading .. ", bigdataset_aurl)
urllib.request.urlretrieve(bigdataset_aurl, "bigdataset.zip")
print("Downloading finished.")
