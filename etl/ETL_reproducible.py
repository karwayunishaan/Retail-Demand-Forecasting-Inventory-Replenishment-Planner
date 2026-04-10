import pandas as pd
import numpy as np
from scipy.stats import norm
import os

def extract():
    """
    Extracts datasets from local storage.
    """
    calendar_og = pd.read_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\Retail Demand Forecasting & Inventory Replenishment Planner\\calendar.csv')
    inventory_og = pd.read_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\Retail Demand Forecasting & Inventory Replenishment Planner\\inventory_daily.csv')
    purchase_orders_og = pd.read_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\Retail Demand Forecasting & Inventory Replenishment Planner\\purchase_orders.csv')
    sales_daily_og = pd.read_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\Retail Demand Forecasting & Inventory Replenishment Planner\\sales_daily.csv')
    stores_og = pd.read_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\Retail Demand Forecasting & Inventory Replenishment Planner\\stores.csv')
    products_og = pd.read_json('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\Retail Demand Forecasting & Inventory Replenishment Planner\\products.json')
    
    return calendar_og, inventory_og, purchase_orders_og, sales_daily_og, stores_og, products_og

def transform(calendar_og, inventory_og, purchase_orders_og, sales_daily_og, stores_og, products_og):
    """
    Performs data cleaning, feature engineering, and replenishment calculations.
    """
    # Converting date to datetime
    sales_daily_og['date'] = pd.to_datetime(sales_daily_og['date'])
    inventory_og['date'] = pd.to_datetime(inventory_og['date'])
    calendar_og['date'] = pd.to_datetime(calendar_og['date'])
    purchase_orders_og['order_date'] = pd.to_datetime(purchase_orders_og['order_date'])
    purchase_orders_og['expected_receipt_date'] = pd.to_datetime(purchase_orders_og['expected_receipt_date'])

    # Creating copies
    sales_daily = sales_daily_og.copy()
    inventory = inventory_og.copy()
    calendar = calendar_og.copy()
    products = products_og.copy()
    purchase_orders = purchase_orders_og.copy()
    stores = stores_og.copy()

    # Defining dates
    label_forecasting_days = 28
    last_date = sales_daily['date'].max()
    as_of_date = last_date - pd.Timedelta(days=label_forecasting_days)
    feature_date = as_of_date - pd.Timedelta(weeks=8)

    # Standardizing datasets
    stores['region'] = stores['region'].str.lower()
    stores['store_size'] = stores['store_size'].str.lower()
    products['category'] = products['category'].str.lower()

    # Handling duplicates and nulls
    purchase_orders = purchase_orders.drop_duplicates()
    inventory['on_hand_close'] = inventory['on_hand_close'].fillna(0)
    sales_daily = sales_daily.drop_duplicates()
    # Calculate the 99.85th percentile (to target that 0.15% spike mention)
    threshold = sales_daily['true_demand_units'].quantile(0.9985)

    # Clip the values so no single day has a massive distortion
    sales_daily['true_demand_units'] = sales_daily['true_demand_units'].clip(upper=threshold)
    
    # Fact Sales Daily
    fact_sales_store_sku_daily = sales_daily.groupby(['date','store_id','sku_id']).agg({
        'units_sold':'sum',
        'true_demand_units':'sum',
        'stockout_censored_units':'sum',
        'revenue':'sum',
        'margin_proxy':'sum'
    }).reset_index()

    fact_sales_store_sku_daily = fact_sales_store_sku_daily.merge(
        calendar[['date','day_of_week','promo_flag','holiday_flag']], 
        how='left', on='date'
    )

    # Fact Inventory Daily
    inventory['stockout_flag'] = (inventory['on_hand_close']==0).astype(int)
    fact_sales = fact_sales_store_sku_daily.copy()
    fact_sales = fact_sales.sort_values('date')
    fact_sales['avg_28d_demand'] = fact_sales.groupby(['store_id','sku_id'])['true_demand_units'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())

    fact_inventory_store_sku_daily = inventory.merge(
        fact_sales[['date','store_id', 'sku_id','avg_28d_demand']],
        on=['date', 'store_id', 'sku_id'], how='left'
    )
    fact_inventory_store_sku_daily['avg_28d_demand'] = fact_inventory_store_sku_daily['avg_28d_demand'].fillna(0)

    # Days of Cover
    fact_inventory_store_sku_daily['days_of_cover'] = round(
        ((fact_inventory_store_sku_daily['on_hand_close'])/(fact_inventory_store_sku_daily['avg_28d_demand'])), 2
    )
    fact_inventory_store_sku_daily['days_of_cover'] = fact_inventory_store_sku_daily['days_of_cover'].replace([np.inf, -np.inf], 999)

    # Replenishment Plan Inputs
    recent_data = fact_sales[fact_sales['date'] >= last_date - pd.Timedelta(days=56)] 
    replenishment_inputs_store_sku = recent_data.groupby(['store_id','sku_id','date'])['true_demand_units'].sum().reset_index()
    replenishment_inputs_store_sku = replenishment_inputs_store_sku.groupby(['store_id','sku_id']).agg(
        avg_daily_demand = ('true_demand_units','mean'),
        demand_std_dev = ('true_demand_units','std')
    ).reset_index()

    # Lead Time Logic
    purchase_orders_grp = purchase_orders.groupby(['store_id','sku_id'])['lead_time_days'].mean().reset_index()
    replenishment_inputs_store_sku = replenishment_inputs_store_sku.merge(purchase_orders_grp, on=['store_id','sku_id'], how='left')
    
    avg_sku = replenishment_inputs_store_sku.groupby('sku_id')['lead_time_days'].mean()
    replenishment_inputs_store_sku['lead_time_days'] = replenishment_inputs_store_sku['lead_time_days'].fillna(
        replenishment_inputs_store_sku['sku_id'].map(avg_sku)
    )
    global_avg = replenishment_inputs_store_sku['lead_time_days'].mean()
    replenishment_inputs_store_sku['lead_time_days'] = replenishment_inputs_store_sku['lead_time_days'].fillna(global_avg)

    # Service Level and Z-Score
    categories_sl = {'personalcare':0.95,'homecare':0.94,'snacks':0.96,'beverages':0.97,'dairy':0.98,'grocery':0.99}
    def shelf_life_calc(shelf_life_days):
        if shelf_life_days < 40:
            return -0.05
        elif shelf_life_days > 40:
            return -0.02
        return 0

    products['service_level_target'] = products['category'].map(categories_sl)
    products['service_level_target'] = (products['service_level_target'] + products['shelf_life_days'].apply(shelf_life_calc)).clip(0.85, 0.99)
    products['z_score'] = norm.ppf(products['service_level_target'])

    # Merge Product Metrics
    replenishment_inputs_store_sku = replenishment_inputs_store_sku.merge(
        products[['sku_id','category','service_level_target','z_score']], on='sku_id', how='left'
    )

    # Safety Stock, ROP, and ROQ
    replenishment_inputs_store_sku['safety_stock'] = np.sqrt(replenishment_inputs_store_sku['lead_time_days']) * \
                                                     replenishment_inputs_store_sku['demand_std_dev'] * \
                                                     replenishment_inputs_store_sku['z_score']
    
    replenishment_inputs_store_sku['ROP'] = (replenishment_inputs_store_sku['avg_daily_demand'] * \
                                            (replenishment_inputs_store_sku['lead_time_days'])) + \
                                            replenishment_inputs_store_sku['safety_stock']

    latest_inventory = (fact_inventory_store_sku_daily.sort_values('date').groupby(['store_id','sku_id']).tail(1))
    
    replenishment_inputs_store_sku = replenishment_inputs_store_sku.merge(
        latest_inventory[['store_id','sku_id','on_hand_close']], on=['store_id','sku_id'], how='left'
    )

    review_days = replenishment_inputs_store_sku['lead_time_days'].mean()
    replenishment_inputs_store_sku['target_stock'] = (replenishment_inputs_store_sku['avg_daily_demand'] * review_days) + \
                                                     replenishment_inputs_store_sku['safety_stock']

    replenishment_inputs_store_sku['ROQ'] = replenishment_inputs_store_sku['target_stock'] - replenishment_inputs_store_sku['on_hand_close']

    return fact_sales_store_sku_daily, fact_inventory_store_sku_daily, replenishment_inputs_store_sku

def load(fact_sales, fact_inventory, replenishment_plan):
    """
    Saves the curated datasets to CSV.
    """
    if not os.path.exists('curated_dts'):
        os.makedirs('curated_dts')
        
    fact_sales.to_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\sampler\\fact_sales_store_sku_daily.csv', index=False)
    fact_inventory.to_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\sampler\\fact_inventory_store_sku_daily.csv', index=False)
    replenishment_plan.to_csv('C:\\Users\\ishaan karwayun\\OneDrive - Symbiosis International University\\Desktop\\BITSOM\\Retail_demand_and_inventory_planner\\sampler\\replenishment_inputs_store_sku.csv', index=False)
    print("ETL Pipeline completed successfully. Files saved in 'sampler/'.")

if __name__ == "__main__":
    # Execute Pipeline
    raw_data = extract()
    transformed_data = transform(*raw_data)
    load(*transformed_data)