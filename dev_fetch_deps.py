import os
import urllib.request

os.mkdir("tests/test_exe/nlohmann")
url = "https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json.hpp"
dest = "examples/test_executables/nlohmann/json.hpp"
urllib.request.urlretrieve(url, dest)
