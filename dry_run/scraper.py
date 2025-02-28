import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from tqdm import tqdm

# this is Iraq LSMS
URL = "https://microdata.worldbank.org/index.php/catalog/2334/data-dictionary"

# Extract module names and descriptions from website and save to main_df
print("Extracting module names and descriptions...")
response = requests.get(URL)
df_list = pd.read_html(response.content)
df = df_list[0]
df["module_name"] = df["Data file"].apply(lambda x: x.split(" ")[0])
df["module_description"] = df["Data file"].apply(lambda x: (" ").join(x.split("_")[1:]))
df.drop(columns=["Data file"], inplace=True)
main_df = df.copy()


# For every module in LSMS, extract variable names, variable descriptions, and variable types
all_dics = []
print("Extracting variable names, descriptions, and types...")
for i in range(len(main_df)):
    print("Extracting from module {}...".format(main_df["module_name"][i]))
    module = main_df["module_name"][i]
    module_description = main_df["module_description"][i]

    # this is url to a specific module of the Iraq LSMS -- may need modify for other countries if the URL structure is different
    module_url = URL + "/F2?file_name={}".format(module)
    response = requests.get(module_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Search for containers that have the info we want -- which is variable name and link to variable webpage
    var_ids = soup.find_all("a", class_="var-id text-break")
    list_of_dics = []
    for var_id in var_ids:
        list_of_dics.append({"module_name": module, "module_description": module_description, "variable_name": var_id.text, "variable_link": var_id["href"]})

    # For every variable in the module, extract variable description and variable type by going to the variable webpage"
    for i in tqdm(range(len(list_of_dics))):
        dic = list_of_dics[i]
        var_link = dic["variable_link"]
        var_response = requests.get(var_link)
        var_soup = BeautifulSoup(var_response.content, 'html.parser')
        var_data = var_soup.find_all("div", class_="variable-container")
        # find variable description and variable type
        var_description = var_data[0].find("h2").text.strip()
        var_type = var_data[0].find("div", class_="fld-inline sum-stat sum-stat-var_intrvl").text.split(":")[1].strip()
        dic["variable_description"] = var_description
        dic["variable_type"] = var_type
    
    all_dics += list_of_dics

pd.DataFrame(all_dics).to_csv("data/iraq_lsms_variables.csv", index=False)

