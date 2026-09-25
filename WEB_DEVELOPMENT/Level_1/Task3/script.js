// Dom Nodes extraction
const previousTextElement = document.getElementById('previous-operand');
const currentTextElement = document.getElementById('current-operand');
const numButtons = document.querySelectorAll('.num-btn');
const opButtons = document.querySelectorAll('.op-btn');
const clearButton = document.querySelector('[data-action="clear"]');
const deleteButton = document.querySelector('[data-action="delete"]');
const equalsButton = document.getElementById('equals');

let currentOperand = '0';
let previousOperand = '';
let activeOperator = undefined;

// Reset Panel Framework
function clearDisplay() {
    currentOperand = '0';
    previousOperand = '';
    activeOperator = undefined;
}

// Slice trailing digit
function deleteDigit() {
    if (currentOperand === '0') return;
    if (currentOperand.length === 1) {
        currentOperand = '0';
    } else {
        currentOperand = currentOperand.slice(0, -1);
    }
}

// String construction for numbers
function appendNumber(number) {
    if (number === '.' && currentOperand.includes('.')) return;
    if (currentOperand === '0' && number !== '.') {
        currentOperand = number.toString();
    } else {
        currentOperand = currentOperand.toString() + number.toString();
    }
}

// Map requested operator paths
function selectOperator(operator) {
    if (currentOperand === '') return;
    if (previousOperand !== '') {
        computeResult();
    }
    activeOperator = operator;
    previousOperand = currentOperand;
    currentOperand = '';
}

// Math logic calculations parsing standard operational branches
function computeResult() {
    let computation;
    const prev = parseFloat(previousOperand);
    const current = parseFloat(currentOperand);
    
    if (isNaN(prev) || isNaN(current)) return;

    // Standard control pathways matching the core project specification
    if (activeOperator === '+') {
        computation = prev + current;
    } else if (activeOperator === '-') {
        computation = prev - current;
    } else if (activeOperator === '*') {
        computation = prev * current;
    } else if (activeOperator === '/') {
        if (current === 0) {
            computation = "Error"; // Avoid Division by Zero anomalies
        } else {
            computation = prev / current;
        }
    } else if (activeOperator === '%') {
        computation = prev % current;
    } else {
        return;
    }

    currentOperand = computation.toString();
    activeOperator = undefined;
    previousOperand = '';
}

// Update DOM element displays natively
function updateDisplay() {
    currentTextElement.innerText = currentOperand;
    if (activeOperator != null) {
        previousTextElement.innerText = `${previousOperand} ${activeOperator}`;
    } else {
        previousTextElement.innerText = '';
    }
}

// Standard Event Listeners Loop binding element interactions
numButtons.forEach(button => {
    button.addEventListener('click', () => {
        appendNumber(button.innerText);
        updateDisplay();
    });
});

opButtons.forEach(button => {
    button.addEventListener('click', () => {
        const opValue = button.getAttribute('data-operator');
        selectOperator(opValue);
        updateDisplay();
    });
});

equalsButton.addEventListener('click', () => {
    computeResult();
    updateDisplay();
});

clearButton.addEventListener('click', () => {
    clearDisplay();
    updateDisplay();
});

deleteButton.addEventListener('click', () => {
    deleteDigit();
    updateDisplay();
});