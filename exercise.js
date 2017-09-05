var fs = require("fs");
var d3 = require("d3");
var _ = require("lodash");
var assert = require("assert");

function getKeys(object) {
    return Object.keys(object).filter(k => object.hasOwnProperty(k));
}

function getValues(object) {
    return getKeys(object).map(k => +object[k]);
}

function select(dataFrame, columnName) {
    return dataFrame.map(x => x[columnName]);
}

function scale(scalar, x) {
    return x.map(element => scalar * element);
}

function add(lhs, rhs) {
    assert(lhs.length === rhs.length);
    return lhs.map((element, index) => element + rhs[index])
}

function inPlaceAdd(lhs, rhs) {
    assert(lhs.length === rhs.length);
    for (var i = lhs.length - 1; i >= 0; i--) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

function sub(lhs, rhs) {
    assert(lhs.length === rhs.length);
    return lhs.map((element, index) => element - rhs[index])
}

function inPlaceSub(lhs, rhs) {
    assert(lhs.length === rhs.length);
    for (var i = lhs.length - 1; i >= 0; i--) {
        lhs[i] -= rhs[i];
    }
    return lhs;
}

function norm(x) {
    return Math.sqrt(x.map(x => x * x).reduce((a, b) => a + b));
}

function binarize(labels, targetLabel) {
    return labels.map(label => label === targetLabel ? 1 : 0);
}


/**
 * 
 * @param {*} X - Array of arrays where each row corresponds to a digit
 * @param {*} y - Array of strings where each row corresponds to the true label for the digit of same index in X
 * @param {*} n - The numbers of elements (features) for each each digit in X
 * @param {*} predict - The prediction function implemented in STEP 1
 */
function calculateGradient(X, y, n, predict) {
    const m = X.length;
    const gradient = new Array(n).fill(0);

    // [STEP 2] TODO: Implement gradient calculation
    // Instructions: For each row in X (called training sample), use 'predict' to obtain the prediction
    // - If the result is equal to y, the prediction is right and nothing needs to be done for this sample
    // - If the result is '1' but the true label is '0' (y element with corresponding index in X), subtract the
    // training sample from the gradient
    // - If the result is '0' but the true label is '1', add the training sample to the gradient
    // After doing this for all rows (training samples) in  X, return 'gradient'
    // Tip: You can use functions provded above to make things easier: 'add', 'inPlaceAdd', 'sub', 'inPlaceSub' and 'scale'
    return gradient;
}

function train(keys, X, y, calculateGradient, learningRate, maxIterations) {
    const insertBias = function insertBias(x) {
        return [1].concat(x);
    };

    const n = keys.length;
    const Xt = X.map(x => insertBias(x));
    const weights = new Array(n + 1).fill(0);

    /**
     * Given the input array 'x' containing color data about a digit and the neurons array 'weights',
     * the function returns 1 when the summation of the products between elements 'x' and 'weights' is greater
     * than or equal to zero, otherwise, returns 0.
     * 'x' and 'weights' have the same length.
     * E.g.: Give x = [1 3 8] and weights = [0.25 -1 1], predict should return 1
     */
    var predict = function predict(x) {
        // [STEP 1] TODO: Implement the body of predict function
    };

    for (var iteration = 0; iteration < maxIterations; iteration++) {
        const gradient = calculateGradient(Xt, y, weights.length, predict);
        inPlaceAdd(weights, scale(learningRate, gradient));

        const normalizedDelta = norm(gradient) / norm(weights);
        console.log(`Iteration: ${iteration}, Delta: ${normalizedDelta}`);
        if (normalizedDelta < 2 * learningRate) {
            break;
        }
    }

    console.log('Number of iterations: ' + iteration);

    return {
        predict: function (x) {
            return predict(insertBias(x));
        }
    }
}

function trainTestSplit(X, y, testRatio) {
    function fromIndexes(array, indexes) {
        return array.filter((element, index) => indexes.includes(index));
    }
    assert(X.length === y.length);
    const size = X.length;
    const indexes = _.shuffle(_.range(size));
    const cutoff = Math.round(size * testRatio);
    const testIndexes = indexes.slice(0, cutoff);
    const trainIndexes = indexes.slice(cutoff, size);
    return {
        train: {
            X: fromIndexes(X, trainIndexes),
            y: fromIndexes(y, trainIndexes)
        },
        test: {
            X: fromIndexes(X, testIndexes),
            y: fromIndexes(y, testIndexes)
        }
    };
}

function score(model, X, y) {
    const predictions = X.map(x => model.predict(x));
    return predictions.filter((p, idx) => p === y[idx]).length / y.length;
}

var model;
var splitData;

fs.readFile("features.csv", "utf8", function(error, data) {
    if (error) {
        throw error;
    }

    const dataFrame = d3.csvParse(data);
    const keys = getKeys(dataFrame[0]);
    const X = dataFrame.map(x => getValues(x));

    fs.readFile("labels.csv", "utf8", function(error, data) {
        if (error) {
            throw error;
        }

        const y = select(d3.csvParse(data), 'digit');
        const zeroes = binarize(y, '0');
        splitData = trainTestSplit(X, zeroes, 0.2);
        model = train(keys, splitData.train.X, splitData.train.y, calculateGradient, 1e-3, 50);
        console.log(`Score on test data: ${score(model, splitData.test.X, splitData.test.y) * 100}%`);
    });
});
