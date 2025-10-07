// Sample data that can be converted to Excel for testing
const sampleData = [
    { "Product": "Laptop", "Sales": 1200, "Profit": 300, "Region": "North", "Month": "Jan" },
    { "Product": "Mouse", "Sales": 25, "Profit": 10, "Region": "North", "Month": "Jan" },
    { "Product": "Keyboard", "Sales": 75, "Profit": 25, "Region": "South", "Month": "Jan" },
    { "Product": "Monitor", "Sales": 400, "Profit": 120, "Region": "East", "Month": "Jan" },
    { "Product": "Laptop", "Sales": 1500, "Profit": 400, "Region": "South", "Month": "Feb" },
    { "Product": "Mouse", "Sales": 30, "Profit": 12, "Region": "East", "Month": "Feb" },
    { "Product": "Keyboard", "Sales": 80, "Profit": 28, "Region": "West", "Month": "Feb" },
    { "Product": "Monitor", "Sales": 350, "Profit": 100, "Region": "North", "Month": "Feb" },
    { "Product": "Laptop", "Sales": 1100, "Profit": 250, "Region": "West", "Month": "Mar" },
    { "Product": "Mouse", "Sales": 22, "Profit": 8, "Region": "South", "Month": "Mar" }
];

console.log('Sample Data Structure:');
console.log('Fields detected:');
Object.keys(sampleData[0]).forEach(key => {
    const sampleValue = sampleData[0][key];
    const isNumeric = typeof sampleValue === 'number' ||
                    (!isNaN(Number(sampleValue)) && sampleValue !== '' && sampleValue !== null);
    console.log(`- ${key}: ${isNumeric ? 'Measure (Numerical)' : 'Dimension (Categorical)'}`);
});

console.log('\nThis data structure would work perfectly with:');
console.log('- GraphicWalker drag-and-drop interface');
console.log('- AI prompt generation (e.g., "Product, Sales, bar chart")');
console.log('- All chart types: bar, line, pie, scatter');

export default sampleData;