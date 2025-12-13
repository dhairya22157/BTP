import pandas as pd

df = pd.read_csv("ar_gk_test.tsv", sep="\t", nrows=5)
print(df.head())
print(df.columns)
recipes = []
current_recipe = []

for _, row in df.iterrows():
    token, tag = row['token'], row['tag']
    
    # Detect recipe boundary (blank line or special token)
    if pd.isna(token) or token.strip() == "":
        if current_recipe:
            recipes.append(" ".join(current_recipe))
            current_recipe = []
    else:
        current_recipe.append(str(token))
        
# Add the last recipe if it wasn’t followed by a blank line
if current_recipe:
    recipes.append(" ".join(current_recipe))

print(f"Total recipes reconstructed: {len(recipes)}")
print("Example:", recipes[0][:150])
