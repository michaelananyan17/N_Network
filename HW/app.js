// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// A more robust CSV parser to handle malformed data in the provided files
function parseCSV(csvText) {
    // Clean up multi-line records and source tags from the provided files
    const textWithoutSources = csvText.replace(/\\s*/g, '');
    const rawLines = textWithoutSources.split('\n').filter(line => line.trim() !== '');
    
    const processedLines = [];
    let currentLine = '';
    for (const line of rawLines) {
        if (/^\d+,/.test(line.trim())) { // If line starts with a number (PassengerId), it's a new record
            if (currentLine) processedLines.push(currentLine);
            currentLine = line.trim();
        } else { // Otherwise, it's a continuation of the previous line
            currentLine += ' ' + line.trim();
        }
    }
    if (currentLine) processedLines.push(currentLine);

    // Parse the cleaned, single-line records
    const headers = processedLines[0].split(',').map(header => header.trim());
    const dataLines = processedLines.slice(1);

    return dataLines.map(line => {
        // Regex splits by comma, but ignores commas inside parentheses to protect the 'Name' field
        const values = line.split(/,(?![^()]*\))/);
        const obj = {};

        headers.forEach((header, i) => {
            let value = (values[i] || '').trim();

            // Clean up the name field which is wrapped in `"(...)"`
            if (header === 'Name' && value.startsWith('"(') && value.endsWith(')"')) {
                value = value.substring(2, value.length - 2).trim();
            }
            
            obj[header] = value === '' ? null : value;
            
            // Convert numerical values to numbers
            if (!isNaN(obj[header]) && obj[header] !== null) {
                obj[header] = parseFloat(obj[header]);
            }
        });
        return obj;
    });
}


// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) {
                survivalBySex[row.Sex] = { survived: 0, total: 0 };
            }
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) {
                survivalBySex[row.Sex].survived++;
            }
        }
    });
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Charts' },
        sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
        { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
    );
    
    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Survived !== undefined) {
            if (!survivalByPclass[row.Pclass]) {
                survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[row.Pclass].total++;
            if (row.Survived === 1) {
                survivalByPclass[row.Pclass].survived++;
            }
        }
    });
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
    );
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Calculate imputation and standardization values from training data ONCE for efficiency
        const ageValues = trainData.map(row => row.Age).filter(age => age !== null);
        const fareValues = trainData.map(row => row.Fare).filter(fare => fare !== null);
        const embarkedValues = trainData.map(row => row.Embarked).filter(e => e !== null);

        const stats = {
            ageMedian: calculateMedian(ageValues),
            fareMedian: calculateMedian(fareValues),
            embarkedMode: calculateMode(embarkedValues),
            ageStdDev: calculateStdDev(ageValues) || 1,
            fareStdDev: calculateStdDev(fareValues) || 1,
        };
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, stats);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });
        
        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };
        
        testData.forEach(row => {
            const features = extractFeatures(row, stats);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}]</p>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization using pre-calculated stats
function extractFeatures(row, stats) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : stats.ageMedian;
    const fare = row.Fare !== null ? row.Fare : stats.fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : stats.embarkedMode;
    
    // Standardize numerical features
    const standardizedAge = (age - stats.ageMedian) / stats.ageStdDev;
    const standardizedFare = (fare - stats.fareMedian) / stats.fareStdDev;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    
    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
    
    // Add optional family features if enabled
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    
    return features;
}

// Calculate median of an array
function calculateMedian(values) {
    if (values.length === 0) return 0;
    
    values.sort((a, b) => a - b);
    const half = Math.floor(values.length / 2);
    
    if (values.length % 2 === 0) {
        return (values[half - 1] + values[half]) / 2;
    }
    
    return values[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (values.length === 0) return null;
    
    const frequency = {};
    let maxCount = 0;
    let mode = null;
    
    values.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > maxCount) {
            maxCount = frequency[value];
            mode = value;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create a sequential model
    model = tf.sequential();
    
    // Add layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape]
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create a model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    
    try {
        // Split training data into train and validation sets (80/20)
        const [trainFeatures, valFeatures] = tf.split(preprocessedTrainData.features, [Math.floor(preprocessedTrainData.features.shape[0] * 0.8), preprocessedTrainData.features.shape[0] - Math.floor(preprocessedTrainData.features.shape[0] * 0.8)]);
        const [trainLabels, valLabels] = tf.split(preprocessedTrainData.labels, [Math.floor(preprocessedTrainData.labels.shape[0] * 0.8), preprocessedTrainData.labels.shape[0] - Math.floor(preprocessedTrainData.labels.shape[0] * 0.8)]);

        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;

        // Define callbacks
        const visorCallback = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        );

        const statusCallback = {
            onEpochEnd: (epoch, logs) => {
                statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
            }
        };
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: [visorCallback, statusCallback] // Use an array for multiple callbacks
        });
        
        statusDiv.innerHTML += '<p>Training completed!</p>';
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    const predVals = validationPredictions.arraySync();
    const trueVals = validationLabels.arraySync();
    
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;
    
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;
    
    await plotROC(trueVals, predVals);
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    const rocData = [];
    
    for (let i = 0; i <= 100; i++) {
        const threshold = i / 100;
        let tp = 0, fn = 0, fp = 0, tn = 0;
        
        for (let j = 0; j < predictions.length; j++) {
            const prediction = predictions[j] >= threshold ? 1 : 0;
            const actual = trueLabels[j];
            
            if (actual === 1) {
                if (prediction === 1) tp++; else fn++;
            } else {
                if (prediction === 1) fp++; else tn++;
            }
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        rocData.push({ x: fpr, y: tpr });
    }
    
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].x - rocData[i-1].x) * (rocData[i].y + rocData[i-1].y) / 2;
    }
    
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: [rocData], series: ['ROC'] },
        { xLabel: 'False Positive Rate', yLabel: 'True Positive Rate', width: 400, height: 400 }
    );
    
    const metricsDiv = document.getElementById('performance-metrics');
    if (!metricsDiv.innerHTML.includes('AUC')) {
        metricsDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
    }
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train the model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    
    try {
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        testPredictions = model.predict(testFeatures);
        const predValues = testPredictions.arraySync();
        
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i] >= 0.5 ? 1 : 0,
            Probability: predValues[i][0]
        }));
        
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;
        
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.keys(row).forEach(key => {
            const td = document.createElement('td');
            td.textContent = key === 'Probability' ? row[key].toFixed(4) : row[key];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        const predValues = testPredictions.arraySync();
        
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predValues[i] >= 0.5 ? 1 : 0}\n`;
        });
        
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predValues[i][0].toFixed(6)}\n`;
        });
        
        // Create download links
        const submissionBlob = new Blob([submissionCSV], { type: 'text/csv' });
        const submissionUrl = URL.createObjectURL(submissionBlob);
        const submissionLink = document.createElement('a');
        submissionLink.href = submissionUrl;
        submissionLink.download = 'submission.csv';
        
        const probabilitiesBlob = new Blob([probabilitiesCSV], { type: 'text/csv' });
        const probabilitiesUrl = URL.createObjectURL(probabilitiesBlob);
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = probabilitiesUrl;
        probabilitiesLink.download = 'probabilities.csv';
        
        document.body.appendChild(submissionLink);
        submissionLink.click();
        document.body.removeChild(submissionLink);
        
        document.body.appendChild(probabilitiesLink);
        probabilitiesLink.click();
        document.body.removeChild(probabilitiesLink);
        
        await model.save('downloads://titanic-tfjs-model');
        
        statusDiv.innerHTML = `
            <p>Export completed!</p>
            <p>Downloaded: submission.csv (Kaggle submission format)</p>
            <p>Downloaded: probabilities.csv (Prediction probabilities)</p>
            <p>Model saved to browser downloads</p>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}