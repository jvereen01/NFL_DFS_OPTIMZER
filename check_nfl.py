import pandas as pd

# Read the NFL Excel file
df = pd.read_excel('NFL.xlsx')

print("=== NFL Fantasy Sheet Analysis ===")
print(f"Total players: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\n=== TOP PLAYER (#1) ===")
print(df.iloc[0].to_string())
print("\n=== TOP 5 PLAYERS ===")
print(df.head().to_string())