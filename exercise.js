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

function calculateGradient(X, y, n, predict) {
    const m = X.length;
    const gradient = new Array(n).fill(0);

    // TODO: [PASSO 2] Implementar cálculo de gradiente
    // No Perceptron, o gradiente é calculado da seguinte forma:
    // Para cada amostra de treino, calcule a "prediction":
    //     Se a prediction for igual ao valor esperado, NÃO altere o gradiente
    //     Se a prediction for positiva (1), mas o valor esperado for negativo (0), subtraia a amostra do gradiente
    //     Se a prediction for negativa (0), mas o valor esperado for positivo (1), some a amostra ao gradiente
    // Após fazer isso para todas as amostras de treino, retorne o array 'gradient'
    // Dica: Pra Utilize as funções add, inPlaceAdd, sub, inPlaceSub e scale para fazer operações com arrays

    return gradient;
}

function train(keys, X, y, calculateGradient, learningRate, maxIterations) {
    const insertBias = function insertBias(x) {
        return [1].concat(x);
    };

    const n = keys.length;
    const Xt = X.map(x => insertBias(x));
    const weights = new Array(n + 1).fill(0);

    var predict = function predict(x) {
        // TODO: [PASSO 1] Implementar a função que, dado um array de características 'x'
        // e os pesos dos neurônios 'weights', retorne 1 caso 'x' pertença a classe desejada ou 0 caso contrário
        // Para isso, o perceptron faz o somatório dos produtos entre os neurônios e os inputs:
        //     Se a soma for maior que 0, então o perceptron retorna 1 (indicando que x pertence a classe 1)
        //     Caso contrário retorna zero
        // O array x e o array weights possuem o mesmo tamanho.
        // Ex: Dado o input x = [1 3 8] e weights = [0.25 -1 1], predict deveria retornar '1'
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
