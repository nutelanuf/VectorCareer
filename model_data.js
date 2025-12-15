// Данные модели для нейросети

const ALL_SKILLS = ["1с: бухгалтерия", "1с: предприятие 8", "1с:erp", "active directory", "adobe illustrator", "adobe indesign", "adobe photoshop", "android", "apache kafka", "apache maven", "api", "b2b продажи", "big data", "ci/cd", "crm", "data science", "deep learning", "devops", "digital marketing", "django framework", "docker", "e-commerce", "fastapi", "figma", "git", "gitlab", "html", "html5", "ios", "java", "javascript", "kubernetes", "linux", "ml", "ms excel", "ms sql", "numpy", "pandas", "postgresql", "postman", "python", "pytorch", "qa", "rabbitmq", "react", "rest", "rest api", "scikit-learn", "smm", "smm-стратегия", "spring boot", "spring framework", "sql", "swagger", "swift", "typescript", "ui", "uml", "ux", "web-дизайн", "адаптация персонала", "администрирование серверов windows", "активные продажи", "акты сверок", "анализ данных", "анализ конкурентной среды", "анализ целевой аудитории", "аналитика", "аналитическое мышление", "английский язык", "банк-клиент", "бухгалтерская отчетность", "веб-дизайн", "векторная графика", "видеосъемка", "грамотная речь", "графический дизайн", "деловая коммуникация", "деловая переписка", "деловое общение", "делопроизводство", "дизайн интерфейсов", "жизненный цикл проекта", "иллюстрирование", "интеграционное тестирование", "интернет-реклама", "контент-маркетинг", "копирайтинг", "маркетинговые коммуникации", "маркетинговый анализ", "моушн-дизайн", "навыки переговоров", "нагрузочное тестирование", "налоговая отчетность", "написание статей", "написание текстов", "наполнение контентом", "настройка пк", "настройка по", "настройка сетевых подключений", "обработка видео", "обучение и развитие", "организаторские навыки", "основы бухгалтерского учета", "первичная бухгалтерская документация", "первичная документация", "планирование маркетинговых кампаний", "планирование продаж", "планирование рекламных кампаний", "подбор персонала", "полиграфический дизайн", "продвижение бренда", "продуктовые метрики", "прототипирование", "работа в команде", "работа с базами данных", "работа с большим объемом информации", "работа с оргтехникой", "развитие бренда", "развитие продаж", "разработка контент-плана", "разработка маркетинговой стратегии", "расчет заработной платы", "регрессионное тестирование", "ремонт пк", "руководство коллективом", "ручное тестирование", "сбор и анализ информации", "социальные сети", "стратегическое мышление", "стратегическое планирование", "телефонные переговоры", "техническая поддержка", "типографика", "точечный подбор персонала", "точность и внимательность к деталям", "управление командой", "управление проектами", "управленческие навыки", "установка по", "формирование ассортимента", "функциональное тестирование", "электронный документооборот"];

const PROFESSION_NAMES = ["Android разработчик", "Data Scientist", "DevOps инженер", "HR-менеджер", "Java разработчик", "Python разработчик", "SMM специалист", "UX/UI дизайнер", "iOS разработчик", "Аналитик данных", "Бизнес-аналитик", "Бухгалтер", "Веб-разработчик", "Графический дизайнер", "Контент-менеджер", "Копирайтер", "Маркетолог", "Менеджер по продажам", "Менеджер проектов", "Продакт-менеджер", "Сетевой инженер", "Системный администратор", "Системный аналитик", "Специалист технической поддержки", "Тестировщик QA"];


// Простая нейросеть на чистом JavaScript (если нет TensorFlow.js)
class SimpleProfessionNeuralNetwork {
    constructor() {
        this.weights = null;
        this.biases = null;
        this.loaded = false;
    }

    // Загрузка предобученных весов
    async loadWeights() {
        try {
            const response = await fetch('neural_weights.json');
            const data = await response.json();
            this.weights = data.weights;
            this.biases = data.biases;
            this.loaded = true;
            console.log('Веса нейросети загружены');
            return true;
        } catch (error) {
            console.error('Ошибка загрузки весов:', error);
            return false;
        }
    }

    // Функция активации ReLU
    relu(x) {
        return Math.max(0, x);
    }

    // Функция активации Softmax
    softmax(arr) {
        const maxVal = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(exp => exp / sumExps);
    }

    // Предсказание профессий
    predict(userSkills, topK = 5) {
        if (!this.loaded) {
            throw new Error('Веса не загружены');
        }

        // 1. Создаем входной вектор
        const inputVector = new Array(ALL_SKILLS.length + 2).fill(0);

        // Заполняем навыки
        userSkills.forEach(skill => {
            const skillLower = skill.toLowerCase().trim();
            const index = ALL_SKILLS.indexOf(skillLower);
            if (index !== -1) {
                inputVector[index] = 1;
            }
        });

        // Последние 2 значения оставляем 0 (зарплата и опыт по умолчанию)

        // 2. Проходим через слои нейросеты
        // Первый слой: Dense + ReLU
        let layer1 = new Array(this.weights[0][0].length).fill(0);
        for (let j = 0; j < layer1.length; j++) {
            let sum = this.biases[0][j];
            for (let i = 0; i < inputVector.length; i++) {
                sum += inputVector[i] * this.weights[0][i][j];
            }
            layer1[j] = this.relu(sum);
        }

        // Второй слой: Dense + ReLU
        let layer2 = new Array(this.weights[1][0].length).fill(0);
        for (let j = 0; j < layer2.length; j++) {
            let sum = this.biases[1][j];
            for (let i = 0; i < layer1.length; i++) {
                sum += layer1[i] * this.weights[1][i][j];
            }
            layer2[j] = this.relu(sum);
        }

        // Выходной слой: Dense + Softmax
        let output = new Array(this.weights[2][0].length).fill(0);
        for (let j = 0; j < output.length; j++) {
            let sum = this.biases[2][j];
            for (let i = 0; i < layer2.length; i++) {
                sum += layer2[i] * this.weights[2][i][j];
            }
            output[j] = sum;
        }

        // Применяем softmax для получения вероятностей
        const probabilities = this.softmax(output);

        // 3. Форматируем результаты
        const results = [];
        for (let i = 0; i < probabilities.length; i++) {
            results.push({
                profession: PROFESSION_NAMES[i],
                probability: probabilities[i],
                percentage: Math.round(probabilities[i] * 100)
            });
        }

        // 4. Сортируем и возвращаем топ-K
        results.sort((a, b) => b.probability - a.probability);
        return results.slice(0, topK);
    }
}

// Создаем глобальный экземпляр
const simpleNeuralNetwork = new SimpleProfessionNeuralNetwork();

// Функции для использования
function getAllSkillsForNeural() {
    return ALL_SKILLS;
}

function getProfessionNamesForNeural() {
    return PROFESSION_NAMES;
}

// Вспомогательная функция для создания вектора из навыков
function createInputVector(userSkills) {
    const vector = new Array(ALL_SKILLS.length + 2).fill(0);
    userSkills.forEach(skill => {
        const skillLower = skill.toLowerCase().trim();
        const index = ALL_SKILLS.indexOf(skillLower);
        if (index !== -1) {
            vector[index] = 1;
        }
    });
    return vector;
}
