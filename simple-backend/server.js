const express = require('express');
const cors = require('cors');
const multer = require('multer');
const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 5246;

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    cb(null, `${timestamp}_${file.originalname}`);
  }
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel'];
    if (allowedTypes.includes(file.mimetype) || file.originalname.endsWith('.xlsx') || file.originalname.endsWith('.xls')) {
      cb(null, true);
    } else {
      cb(new Error('Only Excel files are allowed'));
    }
  }
});

// In-memory storage for uploaded datasets
let uploadedDatasets = [];

// Middleware
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json());

// Sample data for demo
const sampleData = [
  { region: 'North', sales: 120000, month: 'Jan', product: 'Widget A' },
  { region: 'South', sales: 85000, month: 'Jan', product: 'Widget B' },
  { region: 'East', sales: 95000, month: 'Jan', product: 'Widget A' },
  { region: 'West', sales: 110000, month: 'Jan', product: 'Widget C' },
  { region: 'North', sales: 130000, month: 'Feb', product: 'Widget A' },
  { region: 'South', sales: 90000, month: 'Feb', product: 'Widget B' },
  { region: 'East', sales: 105000, month: 'Feb', product: 'Widget A' },
  { region: 'West', sales: 115000, month: 'Feb', product: 'Widget C' },
];

// Field analysis and categorization functions
function analyzeFieldTypes(data) {
  if (!data || data.length === 0) return {};

  const fieldAnalysis = {};
  const sampleRow = data[0];

  Object.keys(sampleRow).forEach(field => {
    const values = data.slice(0, Math.min(50, data.length)).map(row => row[field]).filter(v => v != null);

    if (values.length === 0) {
      fieldAnalysis[field] = { type: 'unknown', category: 'dimension' };
      return;
    }

    // Check if numeric
    const numericValues = values.filter(v => !isNaN(parseFloat(v)) && isFinite(v));
    const isNumeric = numericValues.length > values.length * 0.7;

    // Check if date/time
    const dateValues = values.filter(v => !isNaN(Date.parse(v)));
    const isDate = dateValues.length > values.length * 0.7;

    // Categorize field
    let category = 'dimension';
    let semanticType = 'nominal';
    let analyticType = 'dimension';

    if (isNumeric) {
      category = 'measure';
      semanticType = 'quantitative';
      analyticType = 'measure';
    } else if (isDate) {
      category = 'temporal';
      semanticType = 'temporal';
      analyticType = 'dimension';
    }

    fieldAnalysis[field] = {
      type: isNumeric ? 'numeric' : isDate ? 'date' : 'text',
      category,
      semanticType,
      analyticType,
      uniqueValues: [...new Set(values)].length,
      sampleValues: values.slice(0, 5)
    };
  });

  return fieldAnalysis;
}

function createFieldSynonyms() {
  return {
    // Revenue/Sales synonyms
    revenue: ['revenue', 'sales', 'income', 'earnings', 'proceeds', 'receipts', 'turnover'],
    sales: ['sales', 'revenue', 'income', 'sold', 'amount', 'value'],
    profit: ['profit', 'margin', 'earnings', 'gain', 'return'],
    cost: ['cost', 'expense', 'expenditure', 'spending', 'outlay'],
    price: ['price', 'cost', 'rate', 'fee', 'charge', 'amount'],

    // Quantity synonyms
    quantity: ['quantity', 'qty', 'amount', 'count', 'number', 'volume', 'units'],
    count: ['count', 'number', 'total', 'sum', 'quantity'],

    // Geographic synonyms
    region: ['region', 'area', 'zone', 'territory', 'location', 'place', 'geography'],
    country: ['country', 'nation', 'state'],
    city: ['city', 'town', 'location', 'place'],

    // Product synonyms
    product: ['product', 'item', 'goods', 'merchandise', 'article', 'commodity'],
    category: ['category', 'type', 'class', 'group', 'classification', 'segment'],

    // Time synonyms
    date: ['date', 'time', 'when', 'period'],
    month: ['month', 'monthly', 'mon'],
    year: ['year', 'yearly', 'annual'],
    quarter: ['quarter', 'quarterly', 'qtr'],

    // People synonyms
    customer: ['customer', 'client', 'buyer', 'user', 'consumer'],
    employee: ['employee', 'staff', 'worker', 'person', 'user'],

    // Business synonyms
    department: ['department', 'dept', 'division', 'unit', 'team'],
    channel: ['channel', 'source', 'medium', 'platform', 'method']
  };
}

function matchFieldToPrompt(prompt, availableFields, fieldAnalysis) {
  const lower = prompt.toLowerCase();
  const synonyms = createFieldSynonyms();
  const matches = {};

  // Create reverse lookup for synonyms
  const reverseMap = {};
  Object.entries(synonyms).forEach(([key, values]) => {
    values.forEach(synonym => {
      reverseMap[synonym] = key;
    });
  });

  // Try to match prompt words to actual field names
  Object.keys(availableFields).forEach(fieldName => {
    const fieldLower = fieldName.toLowerCase();
    const fieldWords = fieldLower.split(/[_\s-]+/);

    // Direct field name match
    if (lower.includes(fieldLower)) {
      matches[fieldName] = { confidence: 1.0, matchType: 'direct' };
      return;
    }

    // Partial field word match
    fieldWords.forEach(word => {
      if (lower.includes(word)) {
        matches[fieldName] = { confidence: 0.8, matchType: 'partial' };
      }
    });

    // Synonym matching
    Object.entries(reverseMap).forEach(([synonym, concept]) => {
      if (lower.includes(synonym)) {
        fieldWords.forEach(word => {
          if (synonyms[concept] && synonyms[concept].includes(word)) {
            matches[fieldName] = { confidence: 0.6, matchType: 'synonym' };
          }
        });
      }
    });
  });

  return matches;
}

// Enhanced prompt analysis with context
function analyzePrompt(prompt, dataset = null) {
  const lower = prompt.toLowerCase();

  console.log('analyzePrompt called with:', {
    prompt,
    hasDataset: !!dataset,
    datasetName: dataset ? dataset.datasetName : null,
    hasData: dataset ? !!dataset.data : false,
    hasFieldAnalysis: dataset ? !!dataset.fieldAnalysis : false
  });

  // Use uploaded dataset if available (prioritize any dataset with data)
  if (dataset && dataset.data) {
    // If no field analysis, generate it on the fly
    if (!dataset.fieldAnalysis) {
      console.log('No field analysis found, generating on the fly...');
      dataset.fieldAnalysis = analyzeFieldTypes(dataset.data);
    }
    return analyzePromptWithDataset(prompt, dataset);
  }

  console.log('Using fallback sample data analysis');
  // Fallback to original logic for sample data
  const fields = [];
  if (lower.includes('sales') || lower.includes('revenue')) fields.push('sales');
  if (lower.includes('region')) fields.push('region');
  if (lower.includes('month')) fields.push('month');
  if (lower.includes('product')) fields.push('product');

  // Detect chart type
  let chartType = 'bar';
  if (lower.includes('pie')) chartType = 'pie';
  if (lower.includes('line')) chartType = 'line';
  if (lower.includes('scatter')) chartType = 'scatter';
  if (lower.includes('table')) chartType = 'table';

  const xField = fields.find(f => f !== 'sales') || 'region';
  const yField = 'sales';

  return {
    chartType,
    xField,
    yField,
    fields: fields.length > 0 ? fields : ['region', 'sales'],
    data: sampleData
  };
}

function analyzePromptWithDataset(prompt, dataset) {
  const lower = prompt.toLowerCase();
  const { data, fieldAnalysis } = dataset;

  // Match fields in prompt to actual dataset fields
  const fieldMatches = matchFieldToPrompt(prompt, fieldAnalysis, fieldAnalysis);

  // Get measures and dimensions from field analysis
  const measures = Object.keys(fieldAnalysis).filter(field =>
    fieldAnalysis[field].category === 'measure'
  );
  const dimensions = Object.keys(fieldAnalysis).filter(field =>
    fieldAnalysis[field].category === 'dimension'
  );
  const temporal = Object.keys(fieldAnalysis).filter(field =>
    fieldAnalysis[field].category === 'temporal'
  );

  // Smart field assignment
  let xField = null;
  let yField = null;
  let aggregationType = 'sum'; // Default aggregation

  // Check if the prompt is asking for count
  const isCountQuery = lower.includes('count') || lower.includes('number of');

  // Find best matches for X and Y axes
  const sortedMatches = Object.entries(fieldMatches)
    .sort((a, b) => b[1].confidence - a[1].confidence);

  for (const [fieldName, match] of sortedMatches) {
    const field = fieldAnalysis[fieldName];

    // For count queries, prioritize dimensions for X-axis
    if (!xField && (field.category === 'dimension' || field.category === 'temporal')) {
      xField = fieldName;
    } else if (!yField && field.category === 'measure' && !isCountQuery) {
      yField = fieldName;
    }

    if (xField && (yField || isCountQuery)) break;
  }

  // Handle count queries - use a special count field
  if (isCountQuery) {
    yField = '*'; // Special indicator for count
    aggregationType = 'count';
  }

  // Fallback logic
  if (!yField && measures.length > 0 && !isCountQuery) {
    yField = measures[0];
  }
  if (!xField) {
    if (temporal.length > 0) {
      xField = temporal[0];
    } else if (dimensions.length > 0) {
      xField = dimensions[0];
    }
  }

  // Determine chart type based on data types and prompt
  let chartType = determineChartType(lower, fieldAnalysis[xField], fieldAnalysis[yField]);

  // Generate field definitions for Graphic Walker
  const fieldDefs = Object.keys(fieldAnalysis).map(field => ({
    fid: field,
    semanticType: fieldAnalysis[field].semanticType,
    analyticType: fieldAnalysis[field].analyticType
  }));

  return {
    chartType,
    xField: xField || Object.keys(fieldAnalysis)[0],
    yField: yField || Object.keys(fieldAnalysis)[1],
    fields: Object.keys(fieldAnalysis).map(field => ({
      fid: field,
      semanticType: fieldAnalysis[field].semanticType,
      analyticType: fieldAnalysis[field].analyticType
    })),
    data: data,
    fieldAnalysis,
    fieldMatches,
    availableFields: { measures, dimensions, temporal }
  };
}

function determineChartType(promptLower, xFieldType, yFieldType) {
  // Explicit chart type in prompt
  if (promptLower.includes('pie')) return 'pie';
  if (promptLower.includes('line')) return 'line';
  if (promptLower.includes('scatter')) return 'scatter';
  if (promptLower.includes('table')) return 'table';

  // Smart chart type based on field types
  if (xFieldType && yFieldType) {
    // Time series data
    if (xFieldType.category === 'temporal' && yFieldType.category === 'measure') {
      return 'line';
    }

    // Two numeric fields
    if (xFieldType.category === 'measure' && yFieldType.category === 'measure') {
      return 'scatter';
    }

    // Single dimension for pie chart context
    if (promptLower.includes('distribution') || promptLower.includes('share') ||
        promptLower.includes('percentage') || promptLower.includes('composition')) {
      return 'pie';
    }
  }

  // Default to bar chart
  return 'bar';
}

// API Routes
app.post('/api/prompt/analyze', (req, res) => {
  try {
    const { prompt, datasetName } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    // Find the dataset if specified
    let dataset = null;
    if (datasetName) {
      console.log(`Looking for dataset: ${datasetName}`);
      console.log('Available datasets:', uploadedDatasets.map(d => ({ name: d.datasetName, hasFieldAnalysis: !!d.fieldAnalysis })));
      dataset = uploadedDatasets.find(d => d.datasetName === datasetName);
      if (!dataset) {
        return res.status(404).json({ error: 'Dataset not found' });
      }
      console.log('Found dataset:', {
        name: dataset.datasetName,
        hasData: !!dataset.data,
        hasFieldAnalysis: !!dataset.fieldAnalysis,
        recordCount: dataset.data ? dataset.data.length : 0
      });
    }

    console.log(`Analyzing prompt: "${prompt}" with dataset:`, dataset ? dataset.datasetName : 'none');
    const analysis = analyzePrompt(prompt, dataset);
    console.log('Analysis completed:', analysis ? 'success' : 'failed');

    // Generate field definitions (use analyzed fields if available)
    let fieldDefs;
    if (dataset && dataset.fieldAnalysis) {
      fieldDefs = Object.keys(dataset.fieldAnalysis).map(field => ({
        fid: field,
        semanticType: dataset.fieldAnalysis[field].semanticType,
        analyticType: dataset.fieldAnalysis[field].analyticType
      }));
    } else {
      // Fallback to default fields
      fieldDefs = [
        { fid: 'region', semanticType: 'nominal', analyticType: 'dimension' },
        { fid: 'sales', semanticType: 'quantitative', analyticType: 'measure' },
        { fid: 'month', semanticType: 'nominal', analyticType: 'dimension' },
        { fid: 'product', semanticType: 'nominal', analyticType: 'dimension' }
      ];
    }

    // Generate chart configuration for Graphic Walker
    // Determine the correct encoding based on chart type
    let encodings = {
      dimensions: [],
      measures: [],
      rows: [],
      columns: [],
      color: [],
      opacity: [],
      size: [],
      shape: [],
      radius: [],
      theta: [],
      longitude: [],
      latitude: [],
      geoId: [],
      details: [],
      filters: [],
      text: []
    };

    // Configure encoding based on chart type and field analysis
    if (analysis.chartType === 'bar') {
      // For bar charts: X-axis (dimension), Y-axis (measure)
      if (analysis.xField) {
        encodings.dimensions.push({
          fid: analysis.xField,
          dragId: 'gw_count_fid',
          analyticType: 'dimension'
        });
      }

      // Handle count queries specially
      if (analysis.yField === '*') {
        // For count operations, use any field for counting records
        const anyField = Object.keys(fieldAnalysis)[0];
        if (anyField) {
          encodings.measures.push({
            fid: anyField,
            dragId: 'gw_mea_key_fid',
            analyticType: 'measure',
            aggName: 'count'
          });
        }
      } else if (analysis.yField) {
        encodings.measures.push({
          fid: analysis.yField,
          dragId: 'gw_mea_key_fid',
          analyticType: 'measure'
        });
      }
    } else {
      // Generic fallback
      if (analysis.xField) {
        encodings.dimensions.push({
          fid: analysis.xField,
          dragId: 'gw_count_fid',
          analyticType: 'dimension'
        });
      }

      if (analysis.yField === '*') {
        // For count operations, use any field for counting records
        const anyField = Object.keys(fieldAnalysis)[0];
        if (anyField) {
          encodings.measures.push({
            fid: anyField,
            dragId: 'gw_mea_key_fid',
            analyticType: 'measure',
            aggName: 'count'
          });
        }
      } else if (analysis.yField) {
        encodings.measures.push({
          fid: analysis.yField,
          dragId: 'gw_mea_key_fid',
          analyticType: 'measure'
        });
      }
    }

    // Generate a simpler, more compatible chart configuration for GraphicWalker
    const chartConfig = {
      visId: Date.now().toString(),
      name: `Chart from: "${prompt}"`,
      encodings: {
        dimensions: encodings.dimensions || [],
        measures: encodings.measures || [],
        rows: [],
        columns: [],
        color: [],
        opacity: [],
        size: [],
        shape: [],
        radius: [],
        theta: [],
        longitude: [],
        latitude: [],
        geoId: [],
        details: [],
        filters: [],
        text: []
      },
      config: {
        defaultAggregated: true,
        geoms: [analysis.chartType || 'bar'],
        coordSystem: 'generic',
        stack: 'stack',
        showActions: false,
        interactiveScale: false,
        sorted: 'none',
        zeroScale: true,
        size: { mode: 'fixed', width: 320, height: 200 },
        format: {}
      }
    };

    console.log('Generated chartConfig:', JSON.stringify(chartConfig, null, 2));
    console.log('Analysis result:', { xField: analysis.xField, yField: analysis.yField, chartType: analysis.chartType });

    const responseData = {
      success: true,
      prompt,
      analysis,
      chartConfig,
      fields: fieldDefs,
      data: analysis.data
    };

    // Add debug info if using dataset
    if (dataset) {
      responseData.debug = {
        usedDataset: datasetName,
        fieldMatches: analysis.fieldMatches,
        availableFields: analysis.availableFields,
        fieldAnalysis: analysis.fieldAnalysis
      };
    }

    res.json(responseData);

  } catch (error) {
    console.error('Error analyzing prompt:', error);
    res.status(500).json({ error: 'Failed to analyze prompt' });
  }
});

// Dataset endpoints
app.get('/Dataset', (req, res) => {
  res.json(uploadedDatasets);
});

// Excel upload endpoint
app.post('/Dataset/Upload', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const { datasetName } = req.body;
    if (!datasetName) {
      return res.status(400).json({ error: 'Dataset name is required' });
    }

    // Check if dataset already exists
    if (uploadedDatasets.some(d => d.datasetName === datasetName)) {
      return res.status(400).json({ error: 'Dataset name already exists' });
    }

    // Read Excel file
    const workbook = XLSX.readFile(req.file.path);
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
    const data = XLSX.utils.sheet_to_json(firstSheet, { raw: false, defval: null });

    // Analyze field types and structure
    const fieldAnalysis = analyzeFieldTypes(data);

    // Store dataset info with field analysis
    const dataset = {
      datasetName,
      excelPath: req.file.path,
      isItFromExcel: true,
      sp: '',
      data: data,
      fieldAnalysis: fieldAnalysis,
      uploadedAt: new Date().toISOString(),
      recordCount: data.length,
      fieldCount: Object.keys(fieldAnalysis).length
    };

    uploadedDatasets.push(dataset);

    res.json({
      message: 'File uploaded successfully',
      filePath: req.file.path,
      datasetName
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Failed to upload file' });
  }
});

// Read Excel data endpoint
app.get('/api/excel/read', (req, res) => {
  try {
    const { excelPath } = req.query;

    // Find dataset by path
    const dataset = uploadedDatasets.find(d => d.excelPath === excelPath);
    if (!dataset) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    res.json(dataset.data);
  } catch (error) {
    console.error('Excel read error:', error);
    res.status(500).json({ error: 'Failed to read Excel file' });
  }
});

// Dashboard endpoints
app.get('/Dashboard', (req, res) => {
  res.json([]);
});

app.post('/Dashboard', (req, res) => {
  res.json({ success: true, message: 'Dashboard saved' });
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Simple backend running' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Simple backend running on http://localhost:${PORT}`);
  console.log(`📊 Ready to process prompts for MVP demo!`);
});