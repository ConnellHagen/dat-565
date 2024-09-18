from bs4 import BeautifulSoup
import pandas as pd
import re

filenames = [];
for i in range(1, 41):
    if i < 10:
        filenames += [f"assignment_2/kungalv_slutpriser/kungalv_slutpris_page_0{i}.html"]
    else:
        filenames += [f"assignment_2/kungalv_slutpriser/kungalv_slutpris_page_{i}.html"]

values = []

for filename in filenames:
    with open(filename, 'r', encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    for cell in soup.find_all('li', class_ = 'sold-results__normal-hit'):
        card = cell.a.div

        date_sold = card.find('span', class_ = 'hcl-label--sold-at').text.strip()
        sold_tokens = date_sold.split(' ')
        date_sold = sold_tokens[1] + " " + sold_tokens[2] + " " + sold_tokens[3]

        address = card.find('h2', class_ = 'sold-property-listing__heading').text.strip()

        location = card.find('div', class_ = 'sold-property-listing__location').div.text.strip()[24:]
        location_tokens = location.split(',')

        for i in range (0, len(location_tokens)):
            location_tokens[i] = re.search("(\\w+.*)", location_tokens[i]).group(1)
        location = location_tokens[0]
        for i in range(1, len(location_tokens)):
            location += (", " + location_tokens[i])

        living_area_and_rooms = card.find('div', class_ = 'sold-property-listing__area')
        contains_m2 = living_area_and_rooms.text.strip().find('m²')
        laar_tokens = living_area_and_rooms.text.strip().split('m²')
        living_area = ""


        if len(laar_tokens) == 2:
            living_areas_tokens = laar_tokens[0].split('+')
            if (len(living_areas_tokens) == 1):
                living_area = re.search("(\\d+)", living_areas_tokens[0]).group(1)
            else:
                living_area = re.search("(\\d+)", living_areas_tokens[0]).group(1) + "+" + \
                            re.search("(\\d+).*$", living_areas_tokens[1]).group(1)
            
            rooms = re.search('(\\d+)\xa0rum', laar_tokens[1])
            if (rooms == None):
                rooms = ""
            else:
                rooms = rooms.group(1)

        elif len(laar_tokens) != 2 and contains_m2 != -1:
            living_areas_tokens = laar_tokens[0].split('+')
            if (len(living_areas_tokens) == 1):
                living_area = re.search("(\\d+)", living_areas_tokens[0]).group(1)
            else:
                living_area = re.search("(\\d+)", living_areas_tokens[0]).group(1) + "+" + \
                            re.search("(\\d+).*$", living_areas_tokens[1]).group(1)
        else:
            rooms = re.search('(\\d+)\xa0rum', laar_tokens[0])
            if (rooms == None):
                rooms = ""
            else:
                rooms = rooms.group(1)

        plot_area = card.find('div', class_ = "sold-property-listing__land-area")
        if plot_area == None:
            plot_area = ""
        else:
            plot_area = plot_area.text.strip()[:-8].replace("\xa0", "")

        closing_price = card.find('span', class_ = "hcl-text hcl-text--medium")
        closing_price = closing_price.text.strip()[9:-3].replace("\xa0", "")

        values.append([date_sold, address, location, living_area, rooms, plot_area, closing_price])

data = list()
for val in values:
    row = {
        "Date Sold" : val[0],
        "Address" : val[1],
        "Location" : val[2],
        "Living Area" : val[3],
        "Rooms" : val[4],
        "Plot Area" : val[5],
        "Closing Price" : val[6]
    }
    data.append(row)

data = pd.DataFrame(data)

data.to_csv("assignment_2/housing_data.csv", index = None)
