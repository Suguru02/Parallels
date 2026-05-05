#!/bin/bash

# Устанавливаем флаги компилятора
CXX="g++"
CXXFLAGS="-std=c++17 -pthread -Wall -O2"

# Имя исходного файла (предположим, что он называется server.cpp)
SOURCE="check.cpp"
EXECUTABLE="check"

echo -e "${GREEN}[1/3] Создаю директорию для результатов...${NC}"
mkdir -p "$RESULTS_DIR"

echo -e "${GREEN}[2/3] Компилирую программу...${NC}"
$CXX $CXXFLAGS -o "$EXECUTABLE" "$SOURCE"

if [ $? -ne 0 ]; then
    echo "Ошибка компиляции!"
    exit 1
fi

echo -e "${GREEN}[3/3] Запускаю программу...${NC}"
./"$EXECUTABLE"

echo -e "${GREEN}Готово."