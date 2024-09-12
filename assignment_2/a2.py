from bs4 import BeautifulSoup
import re

filenames = [];
for i in range(1, 41):
    if i < 10:
        filenames += [f"assignment_2/kungalv_slutpriser/kungalv_slutpris_page_0{i}.html"]
    else:
        filenames += [f"assignment_2/kungalv_slutpriser/kungalv_slutpris_page_{i}.html"]

print(filenames)


testfile = "assignment_2/kungalv_slutpriser/kungalv_slutpris_page_01.html"

with open(testfile, 'r', encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

for cell in soup.find_all('li',class_ = 'sold-results__normal-hit'):
    card = cell.a.div

    date_sold = card.find('span', class_ = 'hcl-label--sold-at').text.strip()
    sold_tokens = date_sold.split(' ')
    date_sold = sold_tokens[1] + " " + sold_tokens[2] + " " + sold_tokens[3]

    location = card.find('div', class_ = 'sold-property-listing__location').div.text.strip()[24:]
    location_tokens = location.split(',')
    location_tokens[1] = location_tokens[1][11:]
    location = location_tokens[0] + ", " + location_tokens[1]

    living_area_and_rooms = card.find('div', class_ = 'sold-property-listing__area')
    laar_tokens = living_area_and_rooms.text.strip().split('m²')
    living_area = ""

    living_areas_tokens = laar_tokens[0].split('+')
    if (len(living_areas_tokens) == 1):
        living_area = re.search("(\\d+)", living_areas_tokens[0]).group(1) + "m²"
    else:
        living_area = re.search("(\\d+)", living_areas_tokens[0]).group(1) + "+" + \
                      re.search("(\\d+).*$", living_areas_tokens[1]).group(1) +"m²"
    print(living_area)

    rooms = re.search('(\\d+)\xa0rum', laar_tokens[1]).group(1) + " rum"


  #  num_rooms =

    address = card.find('h2', class_ = 'sold-property-listing__heading').text.strip()


    