# Задание 1  
## Сборка  

### Float  
```  
mkdir build
cd build
cmake .. -DUSE_DOUBLE=OFF
cmake --build . --config Release
./first_task
```  
### Double  
```  
mkdir build
cd build
cmake .. -DUSE_DOUBLE=ON
cmake --build . --config Release
./first_task
```

## Результаты
**Float: Сумма :-0.213894**  

**Double: Сумма :-6.76916e-10**
