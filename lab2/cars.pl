% Раздел 1: факты с 1 аргументом

% Бренды автомобилей (8 фактов)
brand(tesla).
brand(bmw).
brand(toyota).
brand(porsche).
brand(ford).
brand(honda).
brand(audi).
brand(mercedes).

% Типы кузова (6 фактов)
car_type(sedan).
car_type(suv).
car_type(coupe).
car_type(hatchback).
car_type(sports_car).
car_type(truck).

% Типы топлива (4 факта)
fuel_type(electric).
fuel_type(gasoline).
fuel_type(diesel).
fuel_type(hybrid).

% Типы двигателей (5 фактов)
engine_type(v6).
engine_type(v8).
engine_type(inline4).
engine_type(electric_motor).
engine_type(boxer6).

% Типы коробок передач (4 факта)
transmission_type(automatic).
transmission_type(manual).
transmission_type(cvt).
transmission_type(dual_clutch).

% Конкретные автомобили (8 фактов)
car(tesla_model3).
car(bmw_m5).
car(toyota_camry).
car(porsche_911).
car(ford_mustang).
car(honda_civic).
car(audi_a4).
car(mercedes_c63).

% Раздел 2: факты с 2 аргументами

% Марка автомобиля (8 фактов)
car_brand(tesla_model3, tesla).
car_brand(bmw_m5, bmw).
car_brand(toyota_camry, toyota).
car_brand(porsche_911, porsche).
car_brand(ford_mustang, ford).
car_brand(honda_civic, honda).
car_brand(audi_a4, audi).
car_brand(mercedes_c63, mercedes).

% Тип кузова (8 фактов)
car_body_type(tesla_model3, sedan).
car_body_type(bmw_m5, sedan).
car_body_type(toyota_camry, sedan).
car_body_type(porsche_911, sports_car).
car_body_type(ford_mustang, coupe).
car_body_type(honda_civic, hatchback).
car_body_type(audi_a4, sedan).
car_body_type(mercedes_c63, sedan).

% Тип топлива (8 фактов)
car_fuel(tesla_model3, electric).
car_fuel(bmw_m5, gasoline).
car_fuel(toyota_camry, hybrid).
car_fuel(porsche_911, gasoline).
car_fuel(ford_mustang, gasoline).
car_fuel(honda_civic, gasoline).
car_fuel(audi_a4, gasoline).
car_fuel(mercedes_c63, gasoline).

% Двигатель (8 фактов)
car_engine(tesla_model3, electric_motor).
car_engine(bmw_m5, v8).
car_engine(toyota_camry, inline4).
car_engine(porsche_911, boxer6).
car_engine(ford_mustang, v8).
car_engine(honda_civic, inline4).
car_engine(audi_a4, inline4).
car_engine(mercedes_c63, v8).

% Коробка передач (8 фактов)
car_transmission(tesla_model3, automatic).
car_transmission(bmw_m5, automatic).
car_transmission(toyota_camry, automatic).
car_transmission(porsche_911, dual_clutch).
car_transmission(ford_mustang, manual).
car_transmission(honda_civic, cvt).
car_transmission(audi_a4, automatic).
car_transmission(mercedes_c63, automatic).


% Мощность автомобиля (8 фактов)
car_power(tesla_model3, 283).
car_power(bmw_m5, 600).
car_power(toyota_camry, 209).
car_power(porsche_911, 450).
car_power(ford_mustang, 450).
car_power(honda_civic, 158).
car_power(audi_a4, 245).
car_power(mercedes_c63, 510).


% Раздел 3: правила

% Правило 1: Электромобиль - машина с электрическим топливом
electric_car(Car) :- car(Car), car_fuel(Car, electric).

% Правило 2: Спортивная машина - имеет тип кузова sports_car
sports_car(Car) :- car(Car), car_body_type(Car, sports_car).

% Правило 3: Мощная машина - имеет двигатель V8
powerful_car(Car) :- car(Car), car_engine(Car, v8).

% Правило 4: Немецкий автомобиль - марки BMW, Porsche, Audi, Mercedes
german_car(Car) :- car_brand(Car, bmw).
german_car(Car) :- car_brand(Car, porsche).
german_car(Car) :- car_brand(Car, audi).
german_car(Car) :- car_brand(Car, mercedes).

% Правило 5: Машина с механической коробкой
manual_car(Car) :- car(Car), car_transmission(Car, manual).

% Правило 6: Бензиновая спортивная машина
gasoline_sports_car(Car) :- sports_car(Car), car_fuel(Car, gasoline).

% Правило 7: Машина мощнее другой
more_powerful(Car1, Car2) :-
    car_power(Car1, P1),
    car_power(Car2, P2),
    P1 > P2,
    Car1 \== Car2.



