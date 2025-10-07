const XLSX = require('./simple-backend/node_modules/xlsx');

// Create comprehensive business test data
const salesData = [
  { Product_Name: 'Laptop Pro', Monthly_Revenue: 125000, Sales_Region: 'North America', Date: '2024-01-15', Customer_Count: 45 },
  { Product_Name: 'Desktop Elite', Monthly_Revenue: 89000, Sales_Region: 'Europe', Date: '2024-01-15', Customer_Count: 32 },
  { Product_Name: 'Tablet Max', Monthly_Revenue: 156000, Sales_Region: 'Asia Pacific', Date: '2024-01-15', Customer_Count: 78 },
  { Product_Name: 'Phone Ultra', Monthly_Revenue: 234000, Sales_Region: 'North America', Date: '2024-01-15', Customer_Count: 156 },

  { Product_Name: 'Laptop Pro', Monthly_Revenue: 134000, Sales_Region: 'North America', Date: '2024-02-15', Customer_Count: 48 },
  { Product_Name: 'Desktop Elite', Monthly_Revenue: 92000, Sales_Region: 'Europe', Date: '2024-02-15', Customer_Count: 35 },
  { Product_Name: 'Tablet Max', Monthly_Revenue: 167000, Sales_Region: 'Asia Pacific', Date: '2024-02-15', Customer_Count: 82 },
  { Product_Name: 'Phone Ultra', Monthly_Revenue: 245000, Sales_Region: 'North America', Date: '2024-02-15', Customer_Count: 163 },

  { Product_Name: 'Laptop Pro', Monthly_Revenue: 142000, Sales_Region: 'North America', Date: '2024-03-15', Customer_Count: 52 },
  { Product_Name: 'Desktop Elite', Monthly_Revenue: 98000, Sales_Region: 'Europe', Date: '2024-03-15', Customer_Count: 38 },
  { Product_Name: 'Tablet Max', Monthly_Revenue: 178000, Sales_Region: 'Asia Pacific', Date: '2024-03-15', Customer_Count: 87 },
  { Product_Name: 'Phone Ultra', Monthly_Revenue: 256000, Sales_Region: 'North America', Date: '2024-03-15', Customer_Count: 171 }
];

const marketingData = [
  { Campaign_Name: 'Summer Sale', Click_Count: 12500, Conversion_Rate: 3.2, Marketing_Channel: 'Social Media', Campaign_Budget: 25000 },
  { Campaign_Name: 'Back to School', Click_Count: 8900, Conversion_Rate: 4.1, Marketing_Channel: 'Google Ads', Campaign_Budget: 18000 },
  { Campaign_Name: 'Holiday Special', Click_Count: 15600, Conversion_Rate: 2.8, Marketing_Channel: 'Email', Campaign_Budget: 12000 },
  { Campaign_Name: 'New Year Promo', Click_Count: 23400, Conversion_Rate: 5.5, Marketing_Channel: 'Social Media', Campaign_Budget: 35000 },

  { Campaign_Name: 'Spring Launch', Click_Count: 18700, Conversion_Rate: 3.9, Marketing_Channel: 'Google Ads', Campaign_Budget: 28000 },
  { Campaign_Name: 'Valentine Special', Click_Count: 6800, Conversion_Rate: 6.2, Marketing_Channel: 'Email', Campaign_Budget: 8500 },
  { Campaign_Name: 'Easter Deals', Click_Count: 11200, Conversion_Rate: 4.7, Marketing_Channel: 'Social Media', Campaign_Budget: 15000 },
  { Campaign_Name: 'Mother Day Sale', Click_Count: 9500, Conversion_Rate: 5.1, Marketing_Channel: 'Google Ads', Campaign_Budget: 12500 }
];

const employeeData = [
  { Employee_Name: 'John Smith', Annual_Salary: 75000, Department_Name: 'Engineering', Years_Experience: 5, Performance_Score: 4.2 },
  { Employee_Name: 'Sarah Johnson', Annual_Salary: 82000, Department_Name: 'Marketing', Years_Experience: 7, Performance_Score: 4.6 },
  { Employee_Name: 'Mike Wilson', Annual_Salary: 65000, Department_Name: 'Sales', Years_Experience: 3, Performance_Score: 3.8 },
  { Employee_Name: 'Lisa Brown', Annual_Salary: 90000, Department_Name: 'Engineering', Years_Experience: 8, Performance_Score: 4.8 },
  { Employee_Name: 'David Lee', Annual_Salary: 58000, Department_Name: 'Support', Years_Experience: 2, Performance_Score: 4.1 },
  { Employee_Name: 'Jennifer Davis', Annual_Salary: 88000, Department_Name: 'Marketing', Years_Experience: 6, Performance_Score: 4.4 },
  { Employee_Name: 'Robert Taylor', Annual_Salary: 72000, Department_Name: 'Sales', Years_Experience: 4, Performance_Score: 4.0 },
  { Employee_Name: 'Michelle Garcia', Annual_Salary: 95000, Department_Name: 'Engineering', Years_Experience: 10, Performance_Score: 4.9 }
];

// Create workbooks for different business scenarios
console.log('📊 Creating comprehensive business test data...');

// Sales Data Workbook
const salesWb = XLSX.utils.book_new();
const salesWs = XLSX.utils.json_to_sheet(salesData);
XLSX.utils.book_append_sheet(salesWb, salesWs, 'Sales_Data');
XLSX.writeFile(salesWb, './sales-data.xlsx');
console.log('✅ Sales data created: sales-data.xlsx');

// Marketing Data Workbook
const marketingWb = XLSX.utils.book_new();
const marketingWs = XLSX.utils.json_to_sheet(marketingData);
XLSX.utils.book_append_sheet(marketingWb, marketingWs, 'Marketing_Data');
XLSX.writeFile(marketingWb, './marketing-data.xlsx');
console.log('✅ Marketing data created: marketing-data.xlsx');

// HR Data Workbook
const hrWb = XLSX.utils.book_new();
const hrWs = XLSX.utils.json_to_sheet(employeeData);
XLSX.utils.book_append_sheet(hrWb, hrWs, 'Employee_Data');
XLSX.writeFile(hrWb, './employee-data.xlsx');
console.log('✅ Employee data created: employee-data.xlsx');

console.log('🎯 Test scenarios ready:');
console.log('1. Sales Data - Test prompts: "revenue by product", "sales by region", "customers over time"');
console.log('2. Marketing Data - Test prompts: "clicks by channel", "conversion rate by campaign", "budget vs performance"');
console.log('3. Employee Data - Test prompts: "salary by department", "performance vs experience", "headcount by dept"');