# Australian Car Price Analysis and Prediction
![alt text](carco-dealership.jpg)
source: https://www.carco.com.au
## Project Overview
This project started as a personal interest in cars and the intention to buy a car in Australia. Given the complexities of car pricing, I wanted to understand price trends, key factors affecting costs, and the shift towards electric vehicles (EVs). 🚗

To achieve this, I collected and analyzed a dataset of cars in Australia, visualizing the dataset, applying data preprocessing, feature engineering, and machine learning models to extract meaningful insights and create accurate price predictions.📊

## Project goals and objectives
🎯 Goals:

The primary goal of this project is to analyze and predict car prices in the Australian market, providing valuable insights into pricing trends and key factors affecting car values. Through machine learning models, the project aims to improve price prediction accuracy, helping users make informed decisions when purchasing a car. Additionally, it seeks to understand the shift towards electric vehicles (EVs) and their impact on the market. By leveraging data visualization, feature engineering, and advanced models, the project aspires to create a reliable resource for car buyers, researchers, and industry analysts alike.

🔍 Key Objectives:

•	Preprocess the dataset with cleaning and encoding techniques.

•	Create interactive visualizations in Tableau to make insights more  accessible.

•	Analyze the pricing trends of cars based on various attributes.

•	Predict car prices using advanced machine learning models.

•	Investigate the growing electric vehicle (EV) market in Australia.

## About the data

This dataset offers a comprehensive snapshot of the Australian car market with 16,734 rows and 19 features featuring key attributes such as year of manufacture, kilometres driven, engine specifications, fuel consumption, and body type. It also captures car price variations across different brands and locations. This rich dataset provides a perfect foundation to uncover pricing trends, assess the impact of features like fuel type and vehicle condition, and explore how the market is evolving—particularly with the rise of electric vehicles in Australia.

## Dataset Attributes Overview

•	Brand: Name of the car manufacturer (e.g., Toyota, Ford, BMW).

•	Year: The year the car was manufactured, providing insight into the car's age.

•	Model: Specific model name of the car (e.g., Camry, Ranger).

•	Car/Suv: Classification indicating whether the vehicle is a car or SUV.

•	Title: Short description summarizing the car’s main features (e.g., "2021 Toyota Camry Hybrid").

•	UsedOrNew: Condition of the car—whether it is new, used, or a demo vehicle.

•	Transmission: Type of transmission the car has (e.g., Automatic, Manual).

•	Engine: Engine specifications such as engine size or displacement, often measured in liters (e.g., 2.5L).

•	DriveType: Type of drive system (e.g., 4WD, AWD, Front, Rear), indicating traction control on different terrains.

•	FuelType: The type of fuel used by the vehicle (e.g., Petrol, Diesel, Hybrid, Electric).

•	FuelConsumption: Average fuel consumption of the car, typically measured in liters per 100 kilometers (L/100km).

•	Kilometres: The total number of kilometers the car has driven, indicating usage and wear.

•	ColourExtInt: Color of the car's exterior and interior (e.g., "White/Black").

•	CylindersinEngine: Number of cylinders in the engine, which affects performance and efficiency.

•	BodyType: The design style of the vehicle’s body (e.g., Sedan, Hatchback, SUV, UTE).

•	Doors: Number of doors in the car (e.g., 2-door, 4-door).

•	Seats: Number of seats in the car, indicating passenger capacity.

•	Price: Selling price of the car, which is the target variable for price prediction.

## 🛠️ Tools and libraries

•	Python: For data preprocessing, model development, and evaluation.

•	Pandas: For data manipulation and cleaning.

•	NumPy: For numerical operations and efficient calculations.

•	Matplotlib: For plotting charts (e.g., scatter plots and residual distributions).

•	Seaborn: For enhanced statistical visualizations like boxplots and histograms.

•	Scikit-learn: For machine learning models, data splitting, scaling, and evaluation metrics.

•	XGBoost: For high-performance gradient boosting models.

•	LightGBM: For fast and scalable gradient boosting models.

•	Statsmodels: For statistical analysis and assumption testing.

•	Tableau: For interactive dashboards and visual analysis of data trends and distributions.

## Methodology

### Data preproccessing

#### Overview of the Data:

Utilized data.describe() and data.info() methods to understand key statistics and data types across all attributes.

```python
ds = pd.read_csv("Australian Vehicle Prices.csv")

dsdsinfo = ds.info() ds_summary = ds.describe()

print(dsinfo)

print(ds_summary)
```
![alt text](<Screenshot 2025-01-30 150612.png>)

Checked for missing values and overall data consistency using data.

```Python
 ds.isnull().sum() 
 ```

![alt text](<Screenshot 2025-01-30 150621.png>)

Checked for duplicated values and overall data consistency using data.

```Python 
ds_dup = ds.duplicated()
ds_dup
```

![alt text](<Screenshot 2025-01-30 150629.png>)

#### Cleaning Data:
Dropped some unimportant columns:

```Python
ds.drop(['Doors', 'Seats', 'ColourExtInt'],axis=1,inplace=True)
```

Dropped null values of 2 columns and renamed values in the BodyType columm.
"HiAce" → "Hiace"
, "Ute / Trays" → "UTE"
, "Commercial" → "Van"
, "Land" → "Land Rover"


```Python
# Ensure no NaN values in 'Model' and 'Title' columns
ds['Model'] = ds['Model'].fillna('')
ds['Title'] = ds['Title'].fillna('')

# Update 'BodyType' for models starting with "HiAce"
ds.loc[ds['Model'].str.startswith("HIACE"), "BodyType"] = "Van"

# Standardize 'HiAce' to 'Hiace' in the 'Model' and 'Title' columns
ds['Model'] = ds['Model'].str.replace('HiAce', 'Hiace', regex=False)
ds['Title'] = ds['Title'].str.replace('HiAce', 'Hiace', regex=False)

# Verify changes
changed_ds = ds[ds['Model'].str.contains("Hiace")]
changed_ds2 = ds[ds['Title'].str.contains("Hiace")]

changed_ds
```
![alt text](<Screenshot 2025-01-30 160200.png>)

```Python
#Change data values "Commercial", "Van" in "BodyType" column

ds.loc[ds['BodyType'] == "Commercial", "BodyType"] = "Van"

ds[ds['BodyType'] == "Van"].head()
```
![alt text](<Screenshot 2025-01-30 160320.png>)

```Python
ds.loc[ds['BodyType'] == "Ute / Tray", "BodyType"] = "UTE"
```

```Python
# Changing Brand name "Land Rover"

ds.loc[ds['Brand'] == "Land", "Brand"] = "Land Rover"

ds[ds['Brand'] == "Land Rover"].head()
```
Replaced inconsistent name of "Land Rover" model with correct name based on Title

```Python
def categorized_title(title, brand):
    # Return the original title for non-Land Rover cars
    if brand != "Land Rover":  
        return title

    # Check for missing values
    if pd.isna(title):  
        return "Not Found"
        
    # Convert to string to handle non-string values
    title = str(title)  
    if "Evoque" in title:
        return "Evoque"
    elif "Discovery" in title:
        return "Discovery"
    elif "HSE" in title:
        return "HSE"
    elif "Vogue" in title:
        return "Vogue"
    elif "Velar" in title:
        return "Velar"
    elif "Autobiography" in title:
        return "Autobiography"
    elif "Sport" in title:
        return "Sport"
    elif "P530" in title:
        return "P530"
    elif "Defender" in title:
        return "Defender"
    elif "Freelander" in title:
        return "Freelander"
    elif "P525" in title:
        return "P525"
    elif "D300" in title:
        return "D300"
    elif "D350" in title:
        return "D350"
    else:
        return "Not Found"

# Apply the function
ds['Model'] = ds.apply(lambda row: categorized_title(row['Title'], row['Brand']), axis=1)

ds[ds['Brand'] == "Land Rover"].head()
```
![alt text](<Screenshot 2025-01-30 161229.png>)

Added one column named "State" to conveniently visualize the location 

```Python
# Check if 'Location' exists in the DataFrame
if 'Location' in ds.columns:
    # Extract the last 3 characters from the 'Location' column and assign to 'States'
    ds['States'] = ds['Location'].str[-3:]
else:
    print("Column 'Location' does not exist.")

if 'Location' in ds.columns:
    # Extract the last 3 characters
    states_values = ds['Location'].str[-3:]
    
    # Get the position of 'Location'
    location_index = ds.columns.get_loc('Location')
    
    # Insert 'States' next to 'Location'
    if 'States' in ds.columns:
        ds.drop(columns=['States'], inplace=True)
    ds.insert(location_index + 1, 'States', states_values)
else:
    print("Column 'Location' does not exist.")
```

"Other" → Assigned appropriate values based on vehicle Model and Title.
```Python
ds.loc[ds['BodyType'] == "Other", "BodyType"] = ""

ds[ds['BodyType'] == "Other"]
```
Filled missing values in the BodyType column based on relevant attributes like Model and Title.
```Python
# Keywords for each body type
bodytype_keywords = {
    "Sedan": ["City", "C-CLASS", "Impreza", "Calais", "3", "5", "7", "A3", "A5", "RS", "190", "Corolla", "WRX", 
              "S-Class", "Skyline", "Axela", "Cruze", "Accord", "Sai", "E-CLASS", "Crown", "Lancer", "Jetta", 
              "S60", "M3", "Prius", "Model", "Camry", "Sonata", "Cerato", "A4", "Commodore", "Altezza", "Falcon", 
              "C320", "HS", "Flying", "6", "Sebring", "Century", "Berlina", "XJ", "300", "508", "Stanza", "120", 
              "2002", "Model 3", "Pulsar", "C200", "Ghibli"],
    "SUV": ["RAV4", "CR-V", "Tucson", "CX-5", "Forester", "Land Cruiser", "Pajero", "Outlander", "X5", "Kuga", "H2", 
            "Rover", "D90", "Evoque", "Discovery", "Landcruiser", "Sport"],
    "Hatchback": ["Yaris", "2", "Focus", "Golf", "Clio", "i30", "Civic", "Fiesta", "Swift", "Micra", "Tiida", 
                  "Fit", "Note", "MK", "Getz", "Ractis"],
    "UTE": ["Hilux", "Ranger", "Navara", "Colorado", "BT-50", "D-Max", "Triton", "Amarok", "Gladiator", "FPV", 
            "GS", "T60", "Ducato"],
    "Van": ["Hiace", "Transit", "Carnival", "Vito", "Caddy", "iLoad", "Every", "Deliver", "G10+", "Crafter", 
            "Sprinters", "Delivers"],
    "Convertible": ["MX-5", "Z4", "718", "Mustang Convertible", "TT Roadster", "Cabrio", "Corvette"],
    "Wagon": ["Passat", "Outback", "Levorg", "V60", "E-Class Wagon", "Commodore Wagon"],
    "People Mover": ["Alphard", "Vellfire", "Mifa"],
    "Truck": ["Atlas", "Trucks", "NMR", "NLS", "NLR", "Hijet"]
}

# Function to determine the body type from model (case insensitive and cleans spaces)
def get_body_type(model):
    model = model.strip().lower()  # Convert model to lowercase and strip spaces
    for body_type, keywords in bodytype_keywords.items():
        if any(keyword.lower() in model for keyword in keywords):
            return body_type
    return "Not Found"  # If no match found

# Ensure 'BodyType' column exists in the dataframe
ds['BodyType'] = ds.get('BodyType', '')

# Fill missing values only where BodyType is blank or NaN
ds['BodyType'] = ds.apply(lambda row: get_body_type(row['Model'])
                          if pd.isna(row['BodyType']) or row['BodyType'] == ''
                          else row['BodyType'], axis=1)

# Display updated rows for validation
print(ds.loc[ds['BodyType'] == 'Not Found', ['Model', 'BodyType']].head())
```
```Python
# Define the models and their respective body types to be updated
model_to_bodytype = {
    "Hijet": "Truck",
    "Alphard": "People Mover",
    "Vellfire": "People Mover",
    "Crafter": "Van",
    "Hiace": "Van"
}

# Function to update the BodyType based on the model
def update_bodytype(row):
    for model, body_type in model_to_bodytype.items():
        if model in row['Model']:
            return body_type
    return row['BodyType']  # Keep the original value if no match is found

# Apply the function to the BodyType column using .loc to avoid SettingWithCopyWarning
ds.loc[:, 'BodyType'] = ds.apply(update_bodytype, axis=1)

# Display some rows to verify the update
print(ds.loc[ds['Model'].str.contains('Hijet|Alphard|Vellfire|Crafter|Hiace', case=False), ['Model', 'BodyType']].head())
```
Changed categorical values from CylindersinEngine and Engine column

```Python
# Remove 'cyl' from CylindersinEngine and convert to numeric using .loc to avoid SettingWithCopyWarning
ds.loc[:, 'CylindersinEngine'] = ds['CylindersinEngine'].str.replace(r'\D', '', regex=True)
ds.loc[:, 'CylindersinEngine'] = pd.to_numeric(ds['CylindersinEngine'], errors='coerce')

# Move 'CylindersinEngine' next to 'Engine'
columns = ds.columns.tolist()
columns.remove('CylindersinEngine')
engine_index = columns.index('Engine')
columns.insert(engine_index + 1, 'CylindersinEngine')
ds = ds[columns]
```

```Python
# Clean 'Engine' column by keeping only the engine size and removing the cylinder count, retaining 'L' for clarity
ds['Engine'] = ds['Engine'].str.extract(r'(\d+\.\d+|\d+)(?:\s*L)', expand=False)

# Extract only the first decimal number and ignore invalid entries
ds['FuelConsumption'] = ds['FuelConsumption'].astype(str).str.extract(r'(\d+\.\d+|\d+)', expand=False)
ds['FuelConsumption'] = pd.to_numeric(ds['FuelConsumption'], errors='coerce').round(1)

# Convert 'Year' column to integer without decimal places
if 'Year' in ds.columns:
    ds['Year'] = pd.to_numeric(ds['Year'], errors='coerce').astype('Int64')
    
ds.head(5)
```
![alt text](<Screenshot 2025-01-30 162604.png>)

### Data visualization
#### *Overview of the Australian car's price*

![alt text](butterfly.png) 
> Average Car Price by State (Left Heat Map):

•	Northern Territory (NT) and Queensland (QLD) show the lowest average car prices, hovering around $32,000.

•	New South Wales (NSW) and Victoria (VIC) exhibit moderate average car prices in the $34,000–$36,000 range.

•	Western Australia (WA) and Northern Territory (NT) have the highest average car prices, exceeding $37,000.

> Car Supply by State (Right Bar Chart):

•	NSW dominates the market with the highest number of both new and used cars, indicating a central hub for vehicle transactions.

•	Queensland (QLD) and Victoria (VIC) also have a significant number of cars available, with a considerable balance between new and used options.

•	Smaller markets such as Tasmania (TAS), Australian Capital Territory (ACT), and Northern Territory (NT) have much lower car availability.



![alt text](usedandnew.png)

> Dominance of Non-Luxury Brands:

In both new and used car markets, brands such as Ford, BMW, and Mazda maintain relatively stable positions, indicating consistent market demand across both categories.

In the used car market, more economy and mid-range brands such as Kia, Mini, and MG feature prominently, suggesting increased affordability and demand for practical options in the second-hand market.

> Luxury Brands Commanding High Prices:

New Cars: Ferrari, McLaren, and Aston Martin have significantly high average prices, with Ferrari leading at $610,000. Other luxury brands like Bentley, Maserati, and Porsche also hold strong premium values.

Used Cars: Lamborghini and Ferrari still command high prices, with Lamborghini leading at $542,440. Rolls-Royce and Bentley also have substantial average prices, reflecting the retained value of high-end luxury brands in the used market.

#### *Australian average car's price by states*

![alt text](statessales1.png) 

- New South Wales

Avg Price: $43,664
> Luxury brands like Porsche and Genesis as well as some high performance sport car such as Covertte from Chevrolet push the price above average, reflecting high demand for premium cars.
- Victoria

Avg Price: $44,868
> High-end brands such as Bentley and Porsche dominate, suggesting strong demand for luxury vehicles in Melbourne.
- Queensland

Avg Price: $38,318
> A diverse market with both affordable options (Ford, Kia) and premium cars like Chevrolet.
- Australia Capital Territory

Avg Price: $36,277
> A balanced market with moderate interest in both luxury (Porsche, Jaguar) and mid-range brands (Hyundai, Honda).

![alt text](Statessales2.png) 

- South Australia

Avg Price: $37,140
> Porsche and BMW influence higher prices, but affordable options like Ford and Kia maintain market balance.
- Western Australia

Avg Price: $39,524
> High-end brands like Porsche, Tesla and Chevrolet sport cars drive prices above average, indicating demand for premium vehicles.
- Northern Territory

Avg Price: $38,042
> Limited market with lower brand diversity; Ram and Toyota suprisingly stand out for higher prices, while Ford and Hyundai offer budget-friendly options.
- Tasmania

Avg Price: $37,224
> A well-balanced market with moderate interest in premium cars like Audi and Land Rover, alongside mid-range brands like Toyota and Hyundai.

#### *Australian car brands dashboard*
![alt text](<Screenshot 2025-01-08 150131.png>) 
- Luxury Brands Dominate High Prices: Lamborghini, Ferrari, and McLaren lead with average prices exceeding $400,000, reflecting Australia's demand for high-end luxury cars.

- Popular Brands in Demand: Toyota stands out with 2,784 listings, making it the most popular brand, followed by Hyundai (1,239) and Holden (1,087), indicating strong consumer preference for reliable, mid-range options.

- Brand Price Variation Across States: Brands like Mercedes-Benz and Holden show significant price differences depending on the state, emphasizing regional price trends and demand variations.

- Affordability Trends: Common brands such as Ford, Hyundai, and Kia maintain moderate pricing, appealing to a broad market segment focused on cost-effective vehicles.

#### *Popular Car Types in top 4 populated State* 
![alt text](<Dashboard 8.png>)
- New South Wales (NSW): SUVs are the most popular body type, with 2,504 cars, followed by Hatchbacks (1,115) and UTEs (941).

- Queensland (QLD): SUVs dominate the market with 1,181 cars, with Hatchbacks (449) and UTEs (479) also being in high demand.

- Victoria (VIC): SUVs lead with 1,626 cars, while Hatchbacks (606) and Sedans (594) are other popular options.

- Western Australia (WA): SUVs maintain popularity at 792 cars, followed by UTEs (326) and Hatchbacks (268).

> SUV cars is consistently favored across all states, reflecting a preference for versatile, spacious vehicles across Australia.

#### *Australian Electric and Petrol Car*

![alt text](<ev cars.png>)
- EV Cars have a significantly higher average price of $102,201. 

- EV cars are limited in numbers, with only 38 in NSW, 30 in Victoria, and sparse presence across other states.

According to the 2018 report by ClimateWorks Australia, EV sales in Australia rose from just 49 vehicles in 2011 to 2,284 by 2017, representing a growing trend in EV adoption​.
![alt text](<Screenshot 2025-02-01 150125.png>)

Despite this growth, the current map shows relatively low numbers of EV cars spread across various Australian states, with New South Wales and Victoria having relatively higher counts (30 and 38 respectively), indicating that adoption is still niche but improving in major states.

> While interest and infrastructure for EVs are increasing, they still have a long way to go in overtaking petrol cars, particularly in price accessibility for lower tier options and statewide coverage

![alt text](<petrol cars.png>) 
- Petrol Cars have an average price of $36,053

- Petrol cars are abundant, with over 6,300 in NSW and large numbers in Victoria (3,842) and Queensland (2,732).

> Petrol cars maintain strong market dominance, suggesting a broader and more affordable range catering to different consumer segments.

#### *The relationship between Price and other factors such as Kilometres and Fuel Consumption*

![alt text](<Dashboard 9.png>)

Price vs Kilometres:

- The plot shows a negative correlation between price and kilometres driven. As the kilometres increase, the price drops significantly, reflecting depreciation with usage.

- High-priced cars with lower kilometres dominate the dataset, while heavily used cars show prices under $100,000 regardless of brand or type.

Price vs Fuel Consumption:

- This plot suggests a slight positive trend between price and fuel consumption. Luxury cars with high engine performance (often consuming more fuel) tend to have higher prices.

- Lower-priced vehicles generally fall in the low-to-medium fuel consumption range, reinforcing the affordability of fuel-efficient models.
## References
Data source: https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices
ClimateWork Australia report: https://www.aph.gov.au/DocumentStore.ashx?id=be4e9b0a-bf39-442f-8acb-9830038f3617&subId=658041
https://arena.gov.au/assets/2018/06/australian-ev-market-study-report.pdf

