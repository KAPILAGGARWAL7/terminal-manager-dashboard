const XLSX = require('./simple-backend/node_modules/xlsx');

// Create test data
const testData = [
  {product: 'Product A', region: 'North', sales: 50000, month: 'Jan'},
  {product: 'Product B', region: 'South', sales: 45000, month: 'Jan'},
  {product: 'Product C', region: 'East', sales: 60000, month: 'Jan'},
  {product: 'Product A', region: 'North', sales: 55000, month: 'Feb'},
  {product: 'Product B', region: 'South', sales: 48000, month: 'Feb'},
  {product: 'Product C', region: 'East', sales: 62000, month: 'Feb'}
];

// Create workbook
const wb = XLSX.utils.book_new();
const ws = XLSX.utils.json_to_sheet(testData);

// Add worksheet to workbook
XLSX.utils.book_append_sheet(wb, ws, 'TestData');

// Write to file
XLSX.writeFile(wb, './test-data.xlsx');

console.log('✅ Test Excel file created: test-data.xlsx');