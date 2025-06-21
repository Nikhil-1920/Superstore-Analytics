import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint 

# Initialize Rich console
console = Console()

def print_header(title, style="bold cyan"):
    """Prints a styled header panel."""
    console.print(Panel(title, style=style, expand=False, border_style="blue"))

# --- SCRIPT START ---
print_header("Analysis of Superstore Dataset: A Richer Output")

# 1. GATHERING DATA
print_header("1. Gathering Data", style="bold green")
df = pd.read_csv('data/superstore-us.csv')
console.print("Successfully loaded 'superstore-us.csv'")

# 2. UNDERSTANDING DATA
print()
print_header("2. Understanding Data", style="bold green")

print_header("Initial DataFrame Head")
console.print(df.head())

print()
print_header("DataFrame Shape")
console.print(f"The DataFrame has [bold yellow]{df.shape[0]}[/bold yellow] rows and [bold yellow]{df.shape[1]}[/bold yellow] columns.")

print()
print_header("DataFrame Info")
# print()
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
console.print(f"[dim]{info_str}[/dim]")

# 3. STATISTICAL ANALYSIS
print_header("3. Statistical Analysis", style="bold green")
print_header("Statistical Summary")
print()
console.print(df.describe())

# 4. DATA CLEANING
print()
print_header("4. Data Cleaning", style="bold green")

console.print("\n[bold magenta]Dropping 'Postal Code' column...[/bold magenta]")
df.drop(columns="Postal Code", inplace=True)
console.print("DataFrame head after dropping 'Postal Code':")
console.print(df.head())

print()
print_header("Unique Values in 'Country' Column")
console.print(df['Country'].unique())
console.print("\n[bold magenta]Country is always 'United States', dropping column...[/bold magenta]")
df.drop(columns="Country", inplace=True)
console.print("DataFrame head after dropping 'Country':")
console.print(df.head())

# 5. HANDLING NULL AND DUPLICATE VALUES
print()
print_header("5. Handling Null and Duplicate Values", style="bold green")

print_header("Sum of Null Values in Each Column")
print()
console.print(df.isna().sum())

print()
print_header("Checking for Duplicate Values")
num_duplicates = df.duplicated().sum()
console.print(f"Number of duplicate rows found: [bold red]{num_duplicates}[/bold red]")

# For a real effect, it should be df.drop_duplicates(inplace=True)
df.drop_duplicates(inplace=True) 
console.print(f"[bold magenta]Dropped {num_duplicates} duplicate rows. New shape: {df.shape}[/bold magenta]")

# 6. EXPLORATORY DATA ANALYSIS
print()
print_header("6. Exploratory Data Analysis (Unique Values)", style="bold green")

unique_info = {
    'Ship Mode': df['Ship Mode'].unique(),
    'Segment': df['Segment'].unique(),
    'City (Count)': df['City'].nunique(),
    'State (Count)': df['State'].nunique(),
    'Region': df['Region'].unique(),
    'Category': df['Category'].unique(),
    'Sub-Category': df['Sub-Category'].unique()
}

print()
table = Table(title="Summary of Unique Values")
table.add_column("Feature", style="cyan")
table.add_column("Values", style="magenta")

for key, value in unique_info.items():
    table.add_row(key, str(value))

console.print(table)

print()
print_header("Correlation of Sales with Other Numerical Variables")
print()
console.print(df.corr(numeric_only=True)['Sales'].sort_values(ascending=False))

console.print("\n\n[bold green]--- Data exploration and cleaning complete ---[/bold green]")
console.print("[bold]Now proceeding to generate visualizations. Plots will open in separate windows.[/bold]")

plt.rcParams['font.family'] = 'Times New Roman'

# 7. VISUALIZATIONS (Sales and Profit Analysis)
print()
print_header("7. Generating Visualizations", style="bold green")
print()

# Sales Analysis based on region
plt.figure(figsize=(8, 6))
sales_region = df.groupby('Region')['Sales'].sum().sort_values().plot.barh(color='skyblue')
plt.title("Total Sales in each Region")
plt.xlabel("Total Sales")
plt.tight_layout()
for container in sales_region.containers:
    sales_region.bar_label(container, fmt='%.2f')
plt.savefig("sales_by_region.png")
console.print("Generated plot: 'sales_by_region.png'")

# Market Share of each Region
plt.figure(figsize=(8, 8))
df.groupby('Region')['Sales'].sum().plot.pie(autopct="%1.1f%%", wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.title("Market Share of each Region")
plt.ylabel('')
plt.savefig("market_share_by_region.png")
console.print("Generated plot: 'market_share_by_region.png'")

# Profit Analysis based on region
plt.figure(figsize=(8, 6))
profit_region = df.groupby('Region')['Profit'].sum().sort_values().plot.barh(color='lightgreen')
plt.title("Total Profit in each Region")
plt.xlabel("Total Profit")
plt.tight_layout()
for container in profit_region.containers:
    profit_region.bar_label(container, fmt='%.2f')
plt.savefig("profit_by_region.png")
console.print("Generated plot: 'profit_by_region.png'")

# Sales vs Profit Analysis
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Region", alpha=0.6)
plt.title("Sales vs. Profit by Region")
plt.xscale('log') 
plt.savefig("sales_vs_profit.png")
console.print("Generated plot: 'sales_vs_profit.png'")

console.print("\n[bold green]--- Generating Additional Sales Analysis Plots ---[/bold green]")

# Sales Analysis based on ship mode
plt.figure(figsize=(8, 6))
df.groupby('Ship Mode')['Sales'].sum().sort_values().plot.barh(color='tan')
plt.title("Total Sales by Ship Mode")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_by_ship_mode.png")
console.print("Generated plot: 'sales_by_ship_mode.png'")

# Sales Analysis based on segment
plt.figure(figsize=(8, 6))
df.groupby('Segment')['Sales'].sum().sort_values().plot.barh(color='orange')
plt.title("Total Sales by Customer Segment")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_by_segment.png")
console.print("Generated plot: 'sales_by_segment.png'")

# Sales Analysis based on top 5 states
plt.figure(figsize=(8, 6))
df.groupby('State')['Sales'].sum().nlargest(5).sort_values().plot.barh(color='khaki')
plt.title("Top 5 States by Sales")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_by_top5_states.png")
console.print("Generated plot: 'sales_by_top5_states.png'")

# Sales Analysis based on category
plt.figure(figsize=(8, 6))
df.groupby('Category')['Sales'].sum().sort_values().plot.barh(color='orchid')
plt.title("Total Sales by Category")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_by_category.png")
console.print("Generated plot: 'sales_by_category.png'")

console.print("\n[bold green]--- Generating Additional Profit Analysis Plots ---[/bold green]")

# Profit Analysis based on sub-category (Top 5)
plt.figure(figsize=(8, 6))
df.groupby('Sub-Category')['Profit'].sum().nlargest(5).sort_values().plot.barh(color='teal')
plt.title("Top 5 Most Profitable Sub-Categories")
plt.xlabel("Total Profit")
plt.tight_layout()
plt.savefig("profit_by_top5_subcategories.png")
console.print("Generated plot: 'profit_by_top5_subcategories.png'")

# Profit Analysis based on sub-category (Bottom 5)
plt.figure(figsize=(8, 6))
df.groupby('Sub-Category')['Profit'].sum().nsmallest(5).sort_values(ascending=False).plot.barh(color='crimson')
plt.title("Top 5 Least Profitable Sub-Categories")
plt.xlabel("Total Profit")
plt.tight_layout()
plt.savefig("profit_by_bottom5_subcategories.png")
console.print("Generated plot: 'profit_by_bottom5_subcategories.png'")

console.print("\n[bold green]--- Generating Overall Relationship Plots ---[/bold green]")

# Line plot of Profit vs. Discount
plt.figure(figsize=(10, 7))
sns.lineplot(data=df, x="Discount", y="Profit")
plt.title("Profit vs. Discount")
plt.savefig("profit_vs_discount_lineplot.png")
console.print("Generated plot: 'profit_vs_discount_lineplot.png'")

# Bar plot of Sales by Region and Category
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x="Region", y="Sales", hue="Category")
plt.title("Sales by Region across Product Categories")
plt.savefig("sales_by_region_and_category.png")
console.print("Generated plot: 'sales_by_region_and_category.png'")

console.print("\n[bold green on white] Script Finished! [/bold green on white]")
plt.show()