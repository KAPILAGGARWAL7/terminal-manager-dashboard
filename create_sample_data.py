#!/usr/bin/env python3
"""
Create sample Excel data for AI Dashboard Evaluation
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Sample manufacturing/logistics data matching the schema you described
def create_sample_data():
    # Generate 1000 sample records
    data = []

    bay_codes = ['LANE01', 'LANE02', 'LANE03', 'LANE04', 'LANE05', 'BAY_A', 'BAY_B', 'BAY_C']
    product_codes = ['210403', '210404', '210405', '310201', '310202', '410501', '410502']

    base_date = datetime(2024, 9, 1)

    for i in range(1000):
        # Generate realistic data
        gross_quantity = random.choice([0, 0, 0] + list(range(50, 1000, 50)))  # Many zeros as mentioned
        flow_rate = round(random.uniform(5, 50), 1)
        bay_code = random.choice(bay_codes)
        product_code = random.choice(product_codes)

        # Generate realistic dates and times
        scheduled_date = base_date + timedelta(days=random.randint(0, 60))
        # Generate random hour in 12-hour format and AM/PM
        hour_24 = random.randint(6, 22)
        hour_12 = hour_24 if hour_24 <= 12 else hour_24 - 12
        am_pm = 'AM' if hour_24 < 12 else 'PM'
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        exit_time_str = f"{hour_12:02d}:{minute:02d}:{second:02d} {am_pm}"
        exit_time = datetime.combine(
            scheduled_date.date(),
            datetime.strptime(exit_time_str, "%I:%M:%S %p").time()
        )
        created_time = exit_time - timedelta(hours=random.randint(1, 8))

        record = {
            'GrossQuantity': gross_quantity,
            'FlowRate': flow_rate,
            'ShipmentCompartmentID': f"COMP-{i+1:04d}-{random.randint(1000,9999)}",
            'BaseProductID': f"PROD-{product_code}-{random.randint(100,999)}",
            'BaseProductCode': product_code,
            'ShipmentID': f"SHIP-{random.randint(10000,99999)}",
            'ShipmentCode': f"{random.randint(100000000,999999999)}",
            'ExitTime': exit_time.strftime('%I:%M:%S %p'),
            'BayCode': bay_code,
            'ScheduledDate': scheduled_date.strftime('%m/%d/%y'),
            'CreatedTime': created_time.strftime('%I:%M:%S %p')
        }
        data.append(record)

    return pd.DataFrame(data)

# Create the sample data
if __name__ == "__main__":
    print("Creating sample manufacturing/logistics data...")

    # Create sample data
    df = create_sample_data()

    # Save to excel-data directory
    excel_dir = os.path.join(os.path.dirname(__file__), "excel-data")
    os.makedirs(excel_dir, exist_ok=True)

    output_file = os.path.join(excel_dir, "sample_manufacturing_data.xlsx")
    df.to_excel(output_file, index=False)

    print(f"SUCCESS: Created sample data: {output_file}")
    print(f"Records: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Ready for evaluation testing!")

    # Show sample data
    print("\nSample records:")
    print(df.head(3).to_string(index=False))