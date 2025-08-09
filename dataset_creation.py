import pandas as pd
import random

# --- Config ---
num_rows = 25000
output_file = "train_search_dataset.csv"
random.seed(42)

# --- Stations (code, full name) ---
stations = [
    ("NDLS", "New Delhi"), ("CNB", "Kanpur Central"), ("LKO", "Lucknow"),
    ("PNBE", "Patna Jn"), ("GAYA", "Gaya Jn"), ("DBG", "Darbhanga"),
    ("SDAH", "Sealdah"), ("KOAA", "Kolkata"), ("HWH", "Howrah"),
    ("NJP", "New Jalpaiguri"), ("SCL", "Silchar"), ("AGTL", "Agartala"),
    ("DBRG", "Dibrugarh"), ("DBRT", "Dibrugarh Town"), ("NTSK", "New Tinsukia"),
    ("LMG", "Lumding Jn"), ("RNY", "Rangiya Jn"), ("APDJ", "Alipurduar Jn"),
    ("NBQ", "New Bongaigaon"), ("DMV", "Dimapur"), ("MXN", "Mariani Jn"),
    ("KOJ", "Kokrajhar"), ("JRH", "Jorhat Town"), ("GHY", "Guwahati"),
    ("CTC", "Cuttack"), ("BBS", "Bhubaneswar"), ("RNC", "Ranchi"),
    ("JBP", "Jabalpur"), ("BPL", "Bhopal"), ("ADI", "Ahmedabad"),
    ("MMCT", "Mumbai Central"), ("BCT", "Mumbai Central"), ("PUNE", "Pune"),
    ("MAS", "Chennai Central"), ("SBC", "Bangalore City"), ("YPR", "Yesvantpur"),
    ("ERS", "Ernakulam Jn"), ("TVC", "Thiruvananthapuram Central"),
    ("MDU", "Madurai"), ("CBE", "Coimbatore"), ("HYB", "Hyderabad Deccan"),
    ("SC", "Secunderabad"), ("VSKP", "Visakhapatnam"), ("BZA", "Vijayawada"),
    ("KGP", "Kharagpur Jn"), ("MYS", "Mysuru"), ("SRR", "Shoranur Jn"),
    ("VSG", "Vasco-da-Gama"), ("NZM", "Hazrat Nizamuddin"), ("DLI", "Delhi Jn"),
    ("LJN", "Lucknow Jn"), ("PURI", "Puri"), ("BKN", "Bikaner"),
    ("RXL", "Raxaul"), ("JAT", "Jammu Tawi"), ("ASR", "Amritsar"),
    ("CAPE", "Kanyakumari"), ("HW", "Haridwar"), ("SHM", "Shalimar"),
    ("DEE", "Delhi Sarai Rohilla"), ("BDTS", "Bandra Terminus"), ("MS", "Chennai Egmore"),
    ("TPTY", "Tirupati"), ("AII", "Ajmer Jn"), ("CSMT", "Chhatrapati Shivaji Maharaj Terminus"),
    ("FZR", "Firozpur Cantt"), ("ROU", "Rourkela"), ("DDN", "Dehradun"),
    ("KCVL", "Kochuveli")  # Missing earlier — now added
]

# --- Classes ---
classes = ["1A", "2A", "3A", "SL", "CC", "2S"]

# --- Fixed route trains ---
fixed_route_trains = {
    "Saraighat Express": ("HWH", "GHY"),
    "Kamrup Express": ("HWH", "DBRG"),
    "Barak Valley Express": ("LMG", "SCL"),
    "Tripura Sundari Express": ("AGTL", "NZM"),
    "Lachit Express": ("GHY", "DBRG"),
    "Kaziranga Express": ("GHY", "DLI"),
    "Silchar Express": ("SCL", "DBRG"),
    "Kanchanjunga Express": ("SDAH", "AGTL"),
    "Brahmaputra Mail": ("DBRG", "DLI"),
    "Dibrugarh Express": ("DBRG", "YPR"),
    "Nilgiri Express": ("MAS", "MDU"),
    "North East Express": ("GHY", "NDLS"),
    "Vivek Express": ("DBRG", "CAPE"),
    "Avadh Assam Express": ("LJN", "DBRG"),
    "Puri Express": ("BBS", "NDLS"),
    "Mizoram Express": ("AGTL", "LKO"),
    "Sealdah Express": ("SDAH", "BKN"),
    "Goa Express": ("VSG", "NZM"),
    "Satyagrah Express": ("RXL", "NDLS"),
    "Andaman Express": ("MAS", "JAT"),
    "Amritsar Express": ("ASR", "TVC"),
    "Tamil Nadu Express": ("MAS", "NDLS"),
    "Kerala Express": ("TVC", "NDLS"),
    "Jhelum Express": ("JAT", "PUNE"),
    "Utkal Express": ("PURI", "HW"),
    "Shalimar Express": ("SHM", "JAT"),
    "Dee Garib Rath": ("DEE", "BDTS"),
    "Vaigai Express": ("MDU", "MS"),
    "Godavari Express": ("VSKP", "HYB"),
    "Falaknuma Express": ("SC", "HWH"),
    "Kalinga Utkal Express": ("PURI", "NZM"),
    "Ajmer Express": ("AII", "SC"),
    "Padmavati Express": ("TPTY", "SC"),
    "Navjeevan Express": ("ADI", "MAS"),
    "Gitanjali Express": ("HWH", "CSMT"),
    "Rourkela Intercity": ("BBS", "ROU"),
    "Ranchi Rajdhani": ("RNC", "NDLS"),
    "Ernakulam Duronto": ("ERS", "NZM"),
    "CSTM Punjab Mail": ("CSMT", "FZR"),
    "Bikaner Express": ("BKN", "KCVL"),
    "Patna Express": ("PNBE", "SC"),
    "Ahmedabad Express": ("ADI", "MAS"),
    "Howrah Mail": ("HWH", "CSMT"),
    "Doon Express": ("DDN", "HWH"),
    "Tirupati Express": ("TPTY", "SC")
}

# --- Variable route categories ---
base_variable_trains = [
    "Rajdhani Express", "Humsafar Express", "Vande Bharat Express",
    "Shatabdi Express", "Duronto Express", "Garib Rath Express",
    "Sampark Kranti Express", "Tejas Express"
]

used_train_nos = set()
used_train_names = set()
train_definitions = []

def get_unique_train_no():
    while True:
        no = random.randint(10000, 99999)
        if no not in used_train_nos:
            used_train_nos.add(no)
            return no

def get_unique_train_name(base_name, src_name=None):
    if base_name in ["Rajdhani Express", "Humsafar Express", "Vande Bharat Express", "Shatabdi Express"]:
        base_name = f"{src_name} {base_name}"
    name = base_name
    suffix = 1
    while name in used_train_names:
        name = f"{base_name} {suffix}"
        suffix += 1
    used_train_names.add(name)
    return name

# --- 1. Generate train definitions ---
for name, (src_code, dest_code) in fixed_route_trains.items():
    try:
        src_name = next(name for code, name in stations if code == src_code)
        dest_name = next(name for code, name in stations if code == dest_code)
    except StopIteration:
        raise ValueError(f"Station code missing in list: {src_code} or {dest_code}")
    train_no = get_unique_train_no()
    full_name = name
    if full_name in used_train_names:
        full_name += f" ({src_name})"
    used_train_names.add(full_name)
    train_definitions.append((train_no, full_name, src_code, src_name, dest_code, dest_name))

for base_name in base_variable_trains:
    count = random.randint(5, 10)
    for _ in range(count):
        while True:
            src, dest = random.sample(stations, 2)
            if src[0] != dest[0]:
                break
        train_no = get_unique_train_no()
        full_name = get_unique_train_name(base_name, src[1])
        train_definitions.append((train_no, full_name, src[0], src[1], dest[0], dest[1]))

while len(train_definitions) < num_rows // 3:
    while True:
        src, dest = random.sample(stations, 2)
        if src[0] != dest[0]:
            break
    base_name = random.choice(["Express", "Superfast Express", "Intercity Express", "Mail Express"])
    full_name = f"{src[1]}-{dest[1]} {base_name}"
    if full_name in used_train_names:
        continue
    train_no = get_unique_train_no()
    used_train_names.add(full_name)
    train_definitions.append((train_no, full_name, src[0], src[1], dest[0], dest[1]))

# --- 2. Generate rows ---
rows = []
while len(rows) < num_rows:
    train = random.choice(train_definitions)
    train_no, train_name, src_code, src_name, dest_code, dest_name = train
    train_class = random.choice(classes)
    punctuality_rate = round(random.uniform(60, 100), 2)
    avg_waitlist = random.randint(0, 500)
    days_before_journey = random.randint(0, 120)
    weekday = random.randint(0, 6)
    month = random.randint(1, 12)

    base_chance = 100 - (avg_waitlist / 6) - (days_before_journey / 3)
    base_chance = max(5, min(99, base_chance + random.uniform(-5, 5)))
    confirmation_chance = round(base_chance, 2)

    rows.append([
        train_no, train_name, src_code, src_name, dest_code, dest_name,
        train_class, days_before_journey, weekday, month,
        punctuality_rate, avg_waitlist, f"{confirmation_chance}%"
    ])

# --- Save to CSV ---
df = pd.DataFrame(rows, columns=[
    "train_no", "train_name", "source_code", "source_name",
    "destination_code", "destination_name", "class",
    "days_before_journey", "weekday", "month",
    "punctuality_rate", "avg_waitlist", "confirmation_chance"
])
df.to_csv(output_file, index=False)
print(f"✅ Generated {len(df)} rows → {output_file}")

# --- Preview ---
print("\n--- Preview (5 random rows) ---")
print(df.sample(5).to_string(index=False))
