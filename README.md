# Machine Learning JS: Desafio #1
O intuito é criar um Perceptron em JS que classifique caracteres manuscritos do Dataset MNIST.

## Instruções
Execute `npm install` para instalar as dependências.

Abra o arquivo `exercise.js` e preencha os trechos indicados por comentários, começando pelo **PASSO 1**.

Após preencher, execute usando `node exercise.js` e veja o resultado no console.

**Dica:** Observe o Delta no console. Ele deverá diminuir a cada iteração, indicando que o aprendizado está convergindo.

Se você fizer corretamente, o programa irá imprimir algo parecido com `Score on test data: 99.27%`.

## Troubleshooting
### Meu score é 0% / Meu delta é NaN
A sua função de calcular gradiente não está retornando um gradiente numérico válido.

### Meu Delta não decresce
É comum o Delta crescer em algumas iterações, mas se ele está subindo a quase toda iteração, o seu cálculo do gradiente deve estar errado.

### Não consigo de jeito nenhum :(
Tudo bem, você pode conferir a versão resolvida no arquivo `answer.js`
