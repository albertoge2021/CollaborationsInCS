import pycountry_convert as pc

countries = ["Angola","Benin","Burkina Faso","Burundi","Central African Republic","Chad","Comoros","Democratic Republic of the Congo","Djibouti","Eritrea","Ethiopia","Gambia","Guinea","Guinea-Bissau","Lesotho","Liberia","Madagascar","Malawi","Mali","Mauritania","Mozambique","Niger","Rwanda","Sao Tome and Principe","Senegal","Sierra Leone",
             "Somalia","South Sudan","Sudan","Togo","Uganda","United Republic of Tanzania", "Zambia", "Afghanistan", "Bangladesh", "Bhutan", "Cambodia", "Laos", "Myanmar", "Nepal", "Timor-Leste", "Yemen", "Haiti", "Kiribati", "Solomon Islands", "Tuvalu"
]

codes = []

for country in countries:
    codes.append(pc.country_name_to_country_alpha2(country))

print(codes)