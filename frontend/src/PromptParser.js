/**
 * PromptParser - Intelligent natural language processing for chart generation
 * Converts user prompts into GraphicWalker chart specifications
 */

export class PromptParser {

  // Chart type keywords and their priorities
  static CHART_TYPES = {
    'bar': ['bar', 'column', 'bars', 'columns'],
    'line': ['line', 'trend', 'time', 'over time', 'timeline'],
    'scatter': ['scatter', 'plot', 'correlation', 'relationship'],
    'pie': ['pie', 'donut', 'proportion', 'percentage', 'share'],
    'area': ['area', 'filled'],
    'histogram': ['histogram', 'distribution', 'frequency']
  };

  // Field semantic type detection
  static QUANTITATIVE_INDICATORS = [
    'count', 'sum', 'total', 'amount', 'value', 'price', 'cost', 'volume',
    'quantity', 'number', 'avg', 'average', 'mean', 'rate', 'percent',
    'latitude', 'longitude', 'id', 'measure', 'metric'
  ];

  static TEMPORAL_INDICATORS = [
    'date', 'time', 'year', 'month', 'day', 'week', 'timestamp',
    'created', 'updated', 'modified'
  ];

  /**
   * Main parsing function - converts prompt to chart configuration
   * @param {string} prompt - User input like "Petrol_company, Type, bar chart"
   * @param {Array} availableFields - Field definitions from data
   * @param {Array} data - Raw data array
   * @returns {Object} Parsed chart configuration
   */
  static parsePrompt(prompt, availableFields = [], data = []) {
    console.log('PromptParser: Parsing prompt:', prompt);
    console.log('Available fields:', availableFields.map(f => f.name));

    // Clean and normalize prompt
    const cleanPrompt = prompt.toLowerCase().trim();
    const parts = cleanPrompt.split(',').map(p => p.trim());

    // Extract chart type from prompt
    const chartType = this.detectChartType(cleanPrompt);
    console.log('Detected chart type:', chartType);

    // Extract field names from prompt
    const requestedFields = this.extractFieldNames(parts, availableFields);
    console.log('Requested fields:', requestedFields);

    // Map fields to appropriate roles (X, Y, Color, etc.)
    const fieldMapping = this.mapFieldsToRoles(requestedFields, chartType, availableFields);
    console.log('Field mapping:', fieldMapping);

    // Generate chart configuration
    const chartConfig = this.generateChartConfig(chartType, fieldMapping, data);

    return {
      success: true,
      chartType,
      fieldMapping,
      chartConfig,
      suggestedTitle: this.generateTitle(requestedFields, chartType)
    };
  }

  /**
   * Detect chart type from prompt keywords
   */
  static detectChartType(prompt) {
    for (const [type, keywords] of Object.entries(this.CHART_TYPES)) {
      for (const keyword of keywords) {
        if (prompt.includes(keyword)) {
          return type;
        }
      }
    }

    // Default chart type selection based on field count
    return 'bar'; // Default to bar chart
  }

  /**
   * Extract field names from prompt parts, matching with available fields
   */
  static extractFieldNames(parts, availableFields) {
    const extractedFields = [];

    for (const part of parts) {
      // Skip chart type keywords
      if (this.isChartTypeKeyword(part)) continue;

      // Direct field name match
      const directMatch = availableFields.find(f =>
        f.name.toLowerCase() === part ||
        f.name.toLowerCase().includes(part) ||
        part.includes(f.name.toLowerCase())
      );

      if (directMatch) {
        extractedFields.push(directMatch);
        continue;
      }

      // Fuzzy matching for field names with underscores/spaces
      const fuzzyMatch = availableFields.find(f => {
        const normalizedField = f.name.toLowerCase().replace(/[_\s]/g, '');
        const normalizedPart = part.replace(/[_\s]/g, '');
        return normalizedField.includes(normalizedPart) || normalizedPart.includes(normalizedField);
      });

      if (fuzzyMatch) {
        extractedFields.push(fuzzyMatch);
      }
    }

    return extractedFields;
  }

  /**
   * Check if a word is a chart type keyword
   */
  static isChartTypeKeyword(word) {
    return Object.values(this.CHART_TYPES).flat().includes(word);
  }

  /**
   * Map extracted fields to chart roles based on chart type and data semantics
   */
  static mapFieldsToRoles(fields, chartType, availableFields) {
    console.log('PromptParser: Mapping fields to roles, fields:', fields.map(f => f.name));

    if (fields.length === 0) return {};

    // Ensure we have unique fields to work with
    const uniqueFields = [];
    const seenFields = new Set();

    for (const field of fields) {
      if (!seenFields.has(field.name)) {
        uniqueFields.push(field);
        seenFields.add(field.name);
      }
    }

    console.log('PromptParser: Unique fields after deduplication:', uniqueFields.map(f => f.name));

    const mapping = {
      x: null,
      y: null,
      color: null,
      size: null,
      row: null,
      column: null
    };

    // If we only have one field, we cannot create meaningful comparisons
    if (uniqueFields.length === 1) {
      console.log('PromptParser: Only one field available, using count aggregation');
      mapping.x = uniqueFields[0];
      mapping.y = null; // This will trigger count aggregation in GraphicWalker
      return Object.fromEntries(
        Object.entries(mapping).filter(([key, value]) => value !== null)
      );
    }

    // Separate quantitative and categorical fields
    const quantFields = uniqueFields.filter(f => f.analyticType === 'measure');
    const categoricalFields = uniqueFields.filter(f => f.analyticType === 'dimension');

    console.log('PromptParser: Quantitative fields:', quantFields.map(f => f.name));
    console.log('PromptParser: Categorical fields:', categoricalFields.map(f => f.name));

    // Smart field assignment ensuring no duplicates
    switch (chartType) {
      case 'bar':
      case 'column':
        // Bar chart: categorical X, quantitative Y
        mapping.x = categoricalFields[0] || uniqueFields[0];

        // Find a different field for Y axis
        if (quantFields.length > 0) {
          mapping.y = quantFields.find(f => f !== mapping.x) || quantFields[0];
        } else {
          mapping.y = uniqueFields.find(f => f !== mapping.x) || uniqueFields[1];
        }

        // Color should be different from both X and Y
        mapping.color = categoricalFields.find(f => f !== mapping.x && f !== mapping.y) ||
                       quantFields.find(f => f !== mapping.x && f !== mapping.y);
        break;

      case 'line':
        // Line chart: preferably time/categorical on X, quantitative on Y
        const timeField = this.findTimeField(uniqueFields);
        mapping.x = timeField || categoricalFields[0] || uniqueFields[0];
        mapping.y = quantFields.find(f => f !== mapping.x) ||
                   uniqueFields.find(f => f !== mapping.x) || uniqueFields[1];
        mapping.color = uniqueFields.find(f => f !== mapping.x && f !== mapping.y);
        break;

      case 'scatter':
        // Scatter: quantitative X and Y
        if (quantFields.length >= 2) {
          mapping.x = quantFields[0];
          mapping.y = quantFields[1];
        } else {
          mapping.x = uniqueFields[0];
          mapping.y = uniqueFields[1] || uniqueFields[0]; // fallback to same if only 1 field
        }
        mapping.color = categoricalFields.find(f => f !== mapping.x && f !== mapping.y);
        break;

      case 'pie':
        // Pie: categorical for slices, quantitative for values
        mapping.x = categoricalFields[0] || uniqueFields[0];
        mapping.y = quantFields.find(f => f !== mapping.x) ||
                   uniqueFields.find(f => f !== mapping.x) || uniqueFields[1];
        break;

      default:
        // Default mapping - ensure different fields
        mapping.x = uniqueFields[0];
        mapping.y = uniqueFields[1] || uniqueFields[0];
        mapping.color = uniqueFields[2];
    }

    // Final validation - ensure X and Y are different
    if (mapping.x && mapping.y && mapping.x.name === mapping.y.name && uniqueFields.length > 1) {
      console.log('PromptParser: Detected same field for X and Y, fixing...');
      mapping.y = uniqueFields.find(f => f.name !== mapping.x.name) || uniqueFields[1];
    }

    console.log('PromptParser: Final mapping:', {
      x: mapping.x?.name,
      y: mapping.y?.name,
      color: mapping.color?.name
    });

    // Filter out null mappings
    return Object.fromEntries(
      Object.entries(mapping).filter(([key, value]) => value !== null)
    );
  }

  /**
   * Find the best quantitative field from available fields
   */
  static findQuantitativeField(fields) {
    return fields.find(f => f.analyticType === 'measure') ||
           fields.find(f => this.QUANTITATIVE_INDICATORS.some(ind =>
             f.name.toLowerCase().includes(ind)));
  }

  /**
   * Find temporal field for time-based charts
   */
  static findTimeField(fields) {
    return fields.find(f => this.TEMPORAL_INDICATORS.some(ind =>
      f.name.toLowerCase().includes(ind)));
  }

  /**
   * Generate GraphicWalker chart specification
   */
  static generateChartConfig(chartType, fieldMapping, data) {
    console.log('PromptParser: Generating chart config for:', chartType, fieldMapping);

    const config = {
      name: `Generated ${chartType} chart`,
      encodings: {
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
      },
      config: {
        coordSystem: 'generic',
        geoms: [this.mapChartTypeToGeom(chartType)],
        defaultAggregated: false,
        stack: 'none',
        showActions: false,
        interactiveScale: false,
        sorted: 'none',
        zeroScale: true,
        size: {
          mode: 'auto',
          width: 320,
          height: 200
        },
        format: {}
      }
    };

    // Map fields to proper GraphicWalker structure
    if (fieldMapping.x) {
      if (fieldMapping.x.analyticType === 'dimension') {
        config.encodings.dimensions.push({
          fid: fieldMapping.x.fid,
          dragId: fieldMapping.x.fid
        });
      } else {
        config.encodings.measures.push({
          fid: fieldMapping.x.fid,
          dragId: fieldMapping.x.fid
        });
      }
      config.encodings.columns.push({
        fid: fieldMapping.x.fid,
        dragId: fieldMapping.x.fid
      });
    }

    if (fieldMapping.y) {
      if (fieldMapping.y.analyticType === 'dimension') {
        config.encodings.dimensions.push({
          fid: fieldMapping.y.fid,
          dragId: fieldMapping.y.fid
        });
      } else {
        config.encodings.measures.push({
          fid: fieldMapping.y.fid,
          dragId: fieldMapping.y.fid
        });
      }
      config.encodings.rows.push({
        fid: fieldMapping.y.fid,
        dragId: fieldMapping.y.fid
      });
    }

    if (fieldMapping.color) {
      config.encodings.color.push({
        fid: fieldMapping.color.fid,
        dragId: fieldMapping.color.fid
      });
      if (fieldMapping.color.analyticType === 'dimension') {
        config.encodings.dimensions.push({
          fid: fieldMapping.color.fid,
          dragId: fieldMapping.color.fid
        });
      } else {
        config.encodings.measures.push({
          fid: fieldMapping.color.fid,
          dragId: fieldMapping.color.fid
        });
      }
    }

    console.log('PromptParser: Generated config:', config);
    return config;
  }

  /**
   * Get optimal aggregation method based on field type and chart type
   */
  static getOptimalAggregation(field, chartType) {
    if (field.analyticType === 'dimension') {
      return 'count';
    }

    // For quantitative fields, choose aggregation based on chart type and field name
    const fieldName = field.name.toLowerCase();

    if (fieldName.includes('count') || fieldName.includes('number') || fieldName.includes('qty')) {
      return 'sum';
    }

    if (fieldName.includes('avg') || fieldName.includes('average') || fieldName.includes('mean')) {
      return 'mean';
    }

    if (fieldName.includes('rate') || fieldName.includes('percentage') || fieldName.includes('percent')) {
      return 'mean';
    }

    // Chart type specific defaults
    switch (chartType) {
      case 'bar':
      case 'column':
        return 'sum';
      case 'line':
        return fieldName.includes('volume') || fieldName.includes('sales') ? 'sum' : 'mean';
      case 'scatter':
        return 'mean';
      case 'pie':
        return 'sum';
      default:
        return 'sum';
    }
  }

  /**
   * Map chart type to GraphicWalker geomType
   */
  static mapChartTypeToGeom(chartType) {
    const mapping = {
      'bar': 'bar',
      'column': 'bar',
      'line': 'line',
      'scatter': 'point',
      'pie': 'arc',
      'area': 'area',
      'histogram': 'bar'
    };
    return mapping[chartType] || 'bar';
  }

  /**
   * Generate a descriptive title for the chart
   */
  static generateTitle(fields, chartType) {
    if (fields.length === 0) return 'Chart';

    // Remove duplicate field names for title
    const uniqueFieldNames = [...new Set(fields.map(f => f.name))];
    console.log('PromptParser: Generating title for unique fields:', uniqueFieldNames);

    let titleText;
    if (uniqueFieldNames.length === 1) {
      // Single field - show as distribution or analysis
      titleText = `${uniqueFieldNames[0]} Analysis`;
    } else {
      // Multiple fields - join with " vs "
      titleText = uniqueFieldNames.join(' vs ');
    }

    const chartTypeDisplay = chartType.charAt(0).toUpperCase() + chartType.slice(1);
    const finalTitle = `${titleText} (${chartTypeDisplay} Chart)`;

    console.log('PromptParser: Generated title:', finalTitle);
    return finalTitle;
  }

  /**
   * Validate if prompt can be processed
   */
  static validatePrompt(prompt, availableFields) {
    if (!prompt || prompt.trim().length === 0) {
      return { valid: false, error: 'Empty prompt' };
    }

    if (!availableFields || availableFields.length === 0) {
      return { valid: false, error: 'No data fields available' };
    }

    const parts = prompt.toLowerCase().split(',').map(p => p.trim());
    const hasFieldNames = parts.some(part =>
      availableFields.some(field =>
        field.name.toLowerCase().includes(part) ||
        part.includes(field.name.toLowerCase())
      )
    );

    if (!hasFieldNames) {
      return {
        valid: false,
        error: 'No matching field names found. Available fields: ' +
               availableFields.map(f => f.name).join(', ')
      };
    }

    return { valid: true };
  }

  /**
   * Get suggestions for chart types based on field types
   */
  static suggestChartTypes(fields) {
    const quantCount = fields.filter(f => f.analyticType === 'measure').length;
    const categCount = fields.filter(f => f.analyticType === 'dimension').length;

    const suggestions = [];

    if (categCount >= 1 && quantCount >= 1) {
      suggestions.push('bar', 'line', 'pie');
    }

    if (quantCount >= 2) {
      suggestions.push('scatter');
    }

    if (fields.some(f => this.findTimeField([f]))) {
      suggestions.push('line', 'area');
    }

    return suggestions.length > 0 ? suggestions : ['bar'];
  }

  /**
   * Generate example prompts based on available data
   */
  static generateExamplePrompts(availableFields) {
    if (!availableFields || availableFields.length === 0) return [];

    const examples = [];
    const categoricalFields = availableFields.filter(f => f.analyticType === 'dimension');
    const quantFields = availableFields.filter(f => f.analyticType === 'measure');

    if (categoricalFields.length > 0 && quantFields.length > 0) {
      examples.push(`${categoricalFields[0].name}, ${quantFields[0].name}, bar chart`);
    }

    if (categoricalFields.length > 1) {
      examples.push(`${categoricalFields[0].name}, ${categoricalFields[1].name}, pie chart`);
    }

    if (quantFields.length > 1) {
      examples.push(`${quantFields[0].name}, ${quantFields[1].name}, scatter plot`);
    }

    const timeField = this.findTimeField(availableFields);
    if (timeField && quantFields.length > 0) {
      examples.push(`${timeField.name}, ${quantFields[0].name}, line chart`);
    }

    return examples.slice(0, 3); // Return top 3 examples
  }
}

export default PromptParser;