/**
 * TextWrapper - Converts simple English phrases to technical field names and chart types
 * No fancy AI processing - just straightforward mapping
 */

export class TextWrapper {

  // Simple English to field name mapping
  static FIELD_MAPPINGS = {
    // Common terms
    'gross quantity': 'GrossQuantity',
    'grossquantity': 'GrossQuantity',
    'quantity': 'GrossQuantity',
    'flow rate': 'FlowRate',
    'flowrate': 'FlowRate',
    'rate': 'FlowRate',
    'shipment compartment': 'ShipmentCompartmentID',
    'compartment': 'ShipmentCompartmentID',
    'base product': 'BaseProductID',
    'product': 'BaseProductID',
    'terminal': 'Terminal',
    'site': 'Terminal',
    'efficiency': 'Efficiency',
    'uptime': 'Uptime',
    'throughput': 'Throughput',
    'quality': 'Quality',
    'sales': 'TotalSales',
    'total sales': 'TotalSales',
    'profit': 'Profit',
    'region': 'Region',
    'fuel type': 'FuelType',
    'fuel': 'FuelType',
    'daily volume': 'DailyVolume',
    'volume': 'DailyVolume',
    'month': 'Month',
    'status': 'Status',
    'site id': 'SiteId',
    'site name': 'SiteName',
    'operating hours': 'OperatingHours',
    'hours': 'OperatingHours'
  };

  // Action words that indicate chart types
  static CHART_TYPE_MAPPINGS = {
    'compare': 'bar chart',
    'comparison': 'bar chart',
    'vs': 'bar chart',
    'versus': 'bar chart',
    'against': 'bar chart',
    'across': 'bar chart',
    'show': 'bar chart',
    'display': 'bar chart',
    'breakdown': 'pie chart',
    'distribution': 'pie chart',
    'share': 'pie chart',
    'proportion': 'pie chart',
    'trend': 'line chart',
    'over time': 'line chart',
    'timeline': 'line chart',
    'progress': 'line chart',
    'relationship': 'scatter plot',
    'correlation': 'scatter plot',
    'plot': 'scatter plot'
  };

  /**
   * Main function to convert simple English to technical query
   * Example: "Compare GrossQuantity across different FlowRate"
   * -> "GrossQuantity, FlowRate, bar chart"
   */
  static convertToTechnicalQuery(simpleEnglish, availableFields = []) {
    console.log('TextWrapper: Converting simple English:', simpleEnglish);
    console.log('TextWrapper: Available fields:', availableFields.map(f => f.name));

    // Clean the input
    const cleanInput = simpleEnglish.toLowerCase().trim();

    // Extract field names from the input
    const extractedFields = this.extractFieldNames(cleanInput, availableFields);
    console.log('TextWrapper: Extracted fields:', extractedFields);

    // Remove duplicates and ensure we have different fields
    const uniqueFields = [...new Set(extractedFields)];
    console.log('TextWrapper: Unique fields after deduplication:', uniqueFields);

    // Determine chart type from action words
    const chartType = this.determineChartType(cleanInput);
    console.log('TextWrapper: Determined chart type:', chartType);

    // Build technical query
    if (uniqueFields.length === 0) {
      return {
        success: false,
        error: 'No valid fields found in your request',
        originalInput: simpleEnglish
      };
    }

    const technicalQuery = uniqueFields.join(', ') + (chartType ? `, ${chartType}` : '');

    return {
      success: true,
      technicalQuery: technicalQuery,
      extractedFields: uniqueFields,
      chartType: chartType,
      originalInput: simpleEnglish
    };
  }

  /**
   * Extract field names from simple English input
   */
  static extractFieldNames(cleanInput, availableFields) {
    const foundFields = [];

    // Check if input looks like technical query format (comma-separated)
    if (cleanInput.includes(',')) {
      const parts = cleanInput.split(',').map(p => p.trim());
      console.log('TextWrapper: Detected comma-separated input, parts:', parts);

      // Try to match each part directly with available fields
      for (const part of parts) {
        // Skip chart type keywords
        if (this.isChartTypeKeyword(part)) continue;

        const matchingField = availableFields.find(f =>
          f.name.toLowerCase() === part ||
          f.fid.toLowerCase() === part ||
          f.name.toLowerCase().includes(part) ||
          part.includes(f.name.toLowerCase())
        );

        if (matchingField && !foundFields.includes(matchingField.name)) {
          foundFields.push(matchingField.name);
        }
      }

      // If we found fields from comma parsing, return them
      if (foundFields.length > 0) {
        console.log('TextWrapper: Found fields from comma parsing:', foundFields);
        return foundFields;
      }
    }

    // Fallback to natural language processing
    console.log('TextWrapper: Using natural language processing');

    // Try predefined mappings
    for (const [englishTerm, technicalField] of Object.entries(this.FIELD_MAPPINGS)) {
      if (cleanInput.includes(englishTerm)) {
        const matchingField = availableFields.find(f =>
          f.name === technicalField ||
          f.fid === technicalField ||
          f.name.toLowerCase() === technicalField.toLowerCase()
        );

        if (matchingField && !foundFields.includes(matchingField.name)) {
          foundFields.push(matchingField.name);
        } else if (!matchingField && !foundFields.includes(technicalField)) {
          foundFields.push(technicalField);
        }
      }
    }

    // Try direct field name matching
    if (availableFields.length > 0) {
      for (const field of availableFields) {
        const fieldNameLower = field.name.toLowerCase();

        if (cleanInput.includes(fieldNameLower)) {
          if (!foundFields.includes(field.name)) {
            foundFields.push(field.name);
          }
        }
      }
    }

    // Remove duplicates and limit for comparisons
    const uniqueFields = [...new Set(foundFields)];

    if (uniqueFields.length > 2 && (cleanInput.includes('compare') || cleanInput.includes('vs') || cleanInput.includes('across'))) {
      return uniqueFields.slice(0, 2);
    }

    return uniqueFields;
  }

  /**
   * Check if a word is a chart type keyword
   */
  static isChartTypeKeyword(word) {
    const chartKeywords = ['bar', 'line', 'pie', 'scatter', 'chart', 'plot', 'donut'];
    return chartKeywords.some(keyword => word.includes(keyword));
  }

  /**
   * Determine chart type from action words in the input
   */
  static determineChartType(cleanInput) {
    // Check for explicit chart type mentions first
    if (cleanInput.includes('bar') || cleanInput.includes('column')) return 'bar chart';
    if (cleanInput.includes('pie') || cleanInput.includes('donut')) return 'pie chart';
    if (cleanInput.includes('line') || cleanInput.includes('trend')) return 'line chart';
    if (cleanInput.includes('scatter') || cleanInput.includes('plot')) return 'scatter plot';

    // Then check for action words that imply chart types
    for (const [actionWord, chartType] of Object.entries(this.CHART_TYPE_MAPPINGS)) {
      if (cleanInput.includes(actionWord)) {
        return chartType;
      }
    }

    // Default to bar chart for comparison-like queries
    return 'bar chart';
  }

  /**
   * Generate simple English examples based on available fields
   */
  static generateSimpleExamples(availableFields = []) {
    const examples = [];

    if (availableFields.length === 0) {
      return [
        "Compare sales across regions",
        "Show volume trends over time",
        "Distribution of fuel types"
      ];
    }

    // Find different types of fields
    const quantFields = availableFields.filter(f => f.analyticType === 'measure');
    const dimFields = availableFields.filter(f => f.analyticType === 'dimension');

    // Generate examples based on available fields
    if (quantFields.length > 0 && dimFields.length > 0) {
      const quantFieldSimple = this.getTechnicalToSimpleMapping(quantFields[0].name);
      const dimFieldSimple = this.getTechnicalToSimpleMapping(dimFields[0].name);
      examples.push(`Compare ${quantFieldSimple} across different ${dimFieldSimple}`);
    }

    if (quantFields.length > 1) {
      const field1Simple = this.getTechnicalToSimpleMapping(quantFields[0].name);
      const field2Simple = this.getTechnicalToSimpleMapping(quantFields[1].name);
      examples.push(`Show relationship between ${field1Simple} and ${field2Simple}`);
    }

    if (dimFields.length > 0) {
      const dimFieldSimple = this.getTechnicalToSimpleMapping(dimFields[0].name);
      examples.push(`Show distribution of ${dimFieldSimple}`);
    }

    return examples.slice(0, 3);
  }

  /**
   * Convert technical field name back to simple English (reverse mapping)
   */
  static getTechnicalToSimpleMapping(technicalField) {
    // Reverse lookup in our mappings
    for (const [simpleText, techField] of Object.entries(this.FIELD_MAPPINGS)) {
      if (techField === technicalField) {
        return simpleText;
      }
    }

    // If no mapping found, convert camelCase to readable text
    return technicalField
      .replace(/([A-Z])/g, ' $1') // Add space before capitals
      .toLowerCase()
      .trim();
  }

  /**
   * Validate if the simple English input can be processed
   */
  static validateSimpleInput(simpleEnglish, availableFields = []) {
    if (!simpleEnglish || simpleEnglish.trim().length === 0) {
      return {
        valid: false,
        error: 'Please enter what you want to visualize',
        suggestions: this.generateSimpleExamples(availableFields)
      };
    }

    const result = this.convertToTechnicalQuery(simpleEnglish, availableFields);

    if (!result.success) {
      return {
        valid: false,
        error: result.error,
        suggestions: this.generateSimpleExamples(availableFields)
      };
    }

    return { valid: true, result: result };
  }
}

export default TextWrapper;