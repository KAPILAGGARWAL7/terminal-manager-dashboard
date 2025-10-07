// Universal data analysis utilities
export const analyzeDataset = (data, fields) => {
  if (!data || !Array.isArray(data) || data.length === 0) return null;

  const analysis = {
    rowCount: data.length,
    columnCount: fields ? fields.length : Object.keys(data[0]).length,
    dataQuality: calculateDataQuality(data),
    columnAnalysis: analyzeColumns(data, fields),
    suggestions: generateChartSuggestions(data, fields),
    insights: generateInsights(data)
  };

  return analysis;
};

export const calculateDataQuality = (data) => {
  if (!data || data.length === 0) return { score: 0, issues: [] };

  const totalCells = data.length * Object.keys(data[0]).length;
  let completeCells = 0;
  let issues = [];

  data.forEach((row, rowIndex) => {
    Object.entries(row).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '' && value !== 'null') {
        completeCells++;
      } else if (rowIndex < 5) { // Only report first few missing values
        issues.push(`Missing ${key} in row ${rowIndex + 1}`);
      }
    });
  });

  const completenessScore = (completeCells / totalCells) * 100;

  // Check for duplicates
  const uniqueRows = new Set(data.map(row => JSON.stringify(row)));
  if (uniqueRows.size < data.length) {
    issues.push(`${data.length - uniqueRows.size} duplicate rows detected`);
  }

  return {
    score: Math.round(completenessScore),
    completeness: Math.round(completenessScore),
    issues: issues.slice(0, 3), // Limit to 3 issues
    duplicates: data.length - uniqueRows.size
  };
};

export const analyzeColumns = (data, fields) => {
  if (!data || data.length === 0) return [];

  const columns = fields || Object.keys(data[0]).map(key => ({ name: key, fid: key }));

  return columns.map(column => {
    const values = data.map(row => row[column.fid || column.name]).filter(v => v !== null && v !== undefined && v !== '');

    const analysis = {
      name: column.name,
      type: detectColumnType(values),
      uniqueCount: new Set(values).size,
      completeness: Math.round((values.length / data.length) * 100),
      stats: calculateColumnStats(values)
    };

    return analysis;
  });
};

export const detectColumnType = (values) => {
  if (values.length === 0) return 'unknown';

  // Check if all values are numbers
  const numericValues = values.filter(v => !isNaN(Number(v)) && v !== '');
  if (numericValues.length > values.length * 0.8) {
    return 'numeric';
  }

  // Check if values look like dates
  const dateValues = values.filter(v => !isNaN(Date.parse(v)));
  if (dateValues.length > values.length * 0.8) {
    return 'datetime';
  }

  // Check if values look like IDs (contain numbers/letters, high uniqueness)
  const uniqueness = new Set(values).size / values.length;
  if (uniqueness > 0.8 && values[0] && typeof values[0] === 'string') {
    return 'identifier';
  }

  return 'categorical';
};

export const calculateColumnStats = (values) => {
  if (values.length === 0) return {};

  const numericValues = values.filter(v => !isNaN(Number(v))).map(Number);

  if (numericValues.length > values.length * 0.5) {
    // Numeric statistics
    const sorted = numericValues.sort((a, b) => a - b);
    return {
      min: Math.min(...numericValues),
      max: Math.max(...numericValues),
      avg: Math.round(numericValues.reduce((a, b) => a + b, 0) / numericValues.length * 100) / 100,
      median: sorted[Math.floor(sorted.length / 2)]
    };
  } else {
    // Categorical statistics
    const frequency = {};
    values.forEach(v => {
      frequency[v] = (frequency[v] || 0) + 1;
    });

    const mostFrequent = Object.entries(frequency).sort(([,a], [,b]) => b - a)[0];

    return {
      mostFrequent: mostFrequent ? mostFrequent[0] : null,
      categories: Object.keys(frequency).length
    };
  }
};

export const generateChartSuggestions = (data, fields) => {
  if (!data || !fields) return [];

  const suggestions = [];
  const numerics = fields.filter(f => f.semanticType === 'quantitative' || detectColumnType(data.map(row => row[f.fid])) === 'numeric');
  const categoricals = fields.filter(f => f.semanticType === 'nominal' || detectColumnType(data.map(row => row[f.fid])) === 'categorical');
  const temporals = fields.filter(f => detectColumnType(data.map(row => row[f.fid])) === 'datetime');

  // Time series suggestions
  if (numerics.length > 0 && temporals.length > 0) {
    suggestions.push({
      title: `${numerics[0].name} Over Time`,
      description: `Show how ${numerics[0].name} changes over ${temporals[0].name}`,
      chartType: 'line',
      xField: temporals[0].fid,
      yField: numerics[0].fid
    });
  }

  // Distribution suggestions
  if (numerics.length > 0) {
    suggestions.push({
      title: `${numerics[0].name} Distribution`,
      description: `See the distribution of ${numerics[0].name} values`,
      chartType: 'histogram',
      field: numerics[0].fid
    });
  }

  // Category breakdown
  if (categoricals.length > 0 && numerics.length > 0) {
    suggestions.push({
      title: `${numerics[0].name} by ${categoricals[0].name}`,
      description: `Compare ${numerics[0].name} across different ${categoricals[0].name}`,
      chartType: 'bar',
      xField: categoricals[0].fid,
      yField: numerics[0].fid
    });
  }

  // Correlation suggestions
  if (numerics.length > 1) {
    suggestions.push({
      title: `${numerics[0].name} vs ${numerics[1].name}`,
      description: `Explore relationship between ${numerics[0].name} and ${numerics[1].name}`,
      chartType: 'scatter',
      xField: numerics[0].fid,
      yField: numerics[1].fid
    });
  }

  return suggestions.slice(0, 4); // Return top 4 suggestions
};

export const generateInsights = (data) => {
  if (!data || data.length === 0) return [];

  const insights = [];

  // Dataset size insight
  if (data.length > 1000) {
    insights.push(`📊 Large dataset with ${data.length.toLocaleString()} records`);
  } else if (data.length < 100) {
    insights.push(`📊 Small dataset with ${data.length} records`);
  }

  // Column insight
  const columnCount = Object.keys(data[0]).length;
  if (columnCount > 10) {
    insights.push(`🗂️ Rich dataset with ${columnCount} attributes`);
  }

  return insights;
};