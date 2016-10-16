// CUIDADO: Você está prestes a olhar a resposta!
// CUIDADO: Você está prestes a olhar a resposta!
// CUIDADO: Você está prestes a olhar a resposta!
// CUIDADO: Você está prestes a olhar a resposta!
// CUIDADO: Você está prestes a olhar a resposta!

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

function dot(lhs, rhs) {
    assert(lhs.length === rhs.length);
    return lhs.map((element, index) => element * rhs[index]).reduce((a, b) => a + b);
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

function calculateGradient(X, y, n, predict) {
    const m = X.length;
    const gradient = new Array(n).fill(0);
    const predictions = X.map(x => predict(x));

    for (var i = 0; i < m; i++) {
        if (y[i] === predictions[i]) {
            continue;
        }
        const x = X[i];
        if (y[i] === 1) {
            inPlaceAdd(gradient, x);
        } else {
            inPlaceSub(gradient, x);
        }
    }

    return gradient;
}

function binaryThreshold(output) {
    return output >= 0 ? 1 : 0;
}

function train(keys, X, y, calculateGradient, learningRate, maxIterations) {
    const insertBias = function insertBias(x) {
        return [1].concat(x);
    };

    const n = keys.length;
    const Xt = X.map(x => insertBias(x));
    const weights = new Array(n + 1).fill(0);

    var predict = function predict(x) {
        const output = dot(x, weights);
        return binaryThreshold(output);
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
