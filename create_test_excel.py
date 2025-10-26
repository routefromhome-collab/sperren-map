import pandas as pd

streets = [
    "Adlerstraße",
    "Bachstraße", 
    "Friedrich-Engels-Allee",
    "Hofaue",
    "Katernberger Straße"
]

df = pd.DataFrame({"STRNAME": streets})
df.to_excel("2.xlsx", index=False, engine='openpyxl')
print("✓ Тестовый файл 2.xlsx создан с 5 улицами")
