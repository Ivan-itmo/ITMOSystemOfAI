from pyswip import Prolog
import sys
import re

prolog = Prolog()

def load_knowledge_base(filename="cars.pl"):
    try:
        prolog.consult(filename)
        print(f"База знаний '{filename}' загружена.")
        return True
    except Exception as e:
        print(f"Не удалось загрузить '{filename}': {e}")
        return False

def query_values(atom):
    try:
        return [str(r['X']) for r in prolog.query(f"{atom}(X)")]
    except Exception:
        return []

def get_car_prop(car, prop):
    try:
        return [str(r['V']) for r in prolog.query(f"{prop}({car}, V)")]
    except Exception:
        return []

def get_all_cars():
    return query_values("car")


def is_german_car(car):
    try:
        return len(list(prolog.query(f"german_car({car})"))) > 0
    except Exception:
        return False

SYNONYMS = {
    # бренд
    'tesla':      ('brand', 'tesla'),
    'тесла':      ('brand', 'tesla'),
    'bmw':        ('brand', 'bmw'),
    'бмв':        ('brand', 'bmw'),
    'toyota':     ('brand', 'toyota'),
    'тойота':     ('brand', 'toyota'),
    'porsche':    ('brand', 'porsche'),
    'порше':      ('brand', 'porsche'),
    'ford':       ('brand', 'ford'),
    'форд':       ('brand', 'ford'),
    'honda':      ('brand', 'honda'),
    'хонда':      ('brand', 'honda'),
    'audi':       ('brand', 'audi'),
    'ауди':       ('brand', 'audi'),
    'mercedes':   ('brand', 'mercedes'),
    'мерседес':   ('brand', 'mercedes'),

    # тип кузова
    'седан':         ('body_type', 'sedan'),
    'внедорожник':   ('body_type', 'suv'),
    'suv':           ('body_type', 'suv'),
    'кроссовер':     ('body_type', 'suv'),
    'купе':          ('body_type', 'coupe'),
    'хэтчбек':       ('body_type', 'hatchback'),
    'хетчбек':       ('body_type', 'hatchback'),
    'спорткар':      ('body_type', 'sports_car'),
    'спортивная':    ('body_type', 'sports_car'),
    'спортивный':    ('body_type', 'sports_car'),
    'грузовик':      ('body_type', 'truck'),

    # топливо
    'электромобиль': ('fuel', 'electric'),
    'электро':       ('fuel', 'electric'),
    'электрическ':   ('fuel', 'electric'),
    'бензин':        ('fuel', 'gasoline'),
    'дизель':        ('fuel', 'diesel'),
    'гибрид':        ('fuel', 'hybrid'),

    # двигатель
    'v6':         ('engine', 'v6'),
    'v8':         ('engine', 'v8'),
    'boxer':      ('engine', 'boxer6'),
    'оппозитн':   ('engine', 'boxer6'),

    # коробка передач
    'автомат':       ('transmission', 'automatic'),
    'механик':       ('transmission', 'manual'),
    'вариатор':      ('transmission', 'cvt'),
    'робот':         ('transmission', 'dual_clutch'),

    # страна
    'немецк':     ('country', 'german'),
    'германи':    ('country', 'german'),
    'американск': ('country', 'american'),
    'сша':        ('country', 'american'),
    'японск':     ('country', 'japanese'),
    'япони':      ('country', 'japanese'),
}


def parse_power(text):
    prefs = []
    text_lower = text.lower()

    for pattern in [r'от\s+(\d+)', r'не\s+менее\s+(\d+)', r'больше\s+(\d+)', r'свыше\s+(\d+)']:
        m = re.search(pattern, text_lower)
        if m:
            prefs.append(('min_power', int(m.group(1))))
            break

    for pattern in [r'до\s+(\d+)', r'не\s+более\s+(\d+)', r'меньше\s+(\d+)']:
        m = re.search(pattern, text_lower)
        if m:
            prefs.append(('max_power', int(m.group(1))))
            break

    return prefs


def parse_input(text):
    prefs = []
    text_lower = text.lower()

    # 1 Ищем синонимы критериев
    for word, (key, val) in SYNONYMS.items():
        if word in text_lower:
            pair = (key, val)
            if pair not in prefs:
                prefs.append(pair)

    # 2 Ищем мощность
    for pair in parse_power(text_lower):
        if pair not in prefs:
            prefs.append(pair)

    return prefs

def get_country(car):
    countries = []
    if is_german_car(car):
        countries.append('german')
    brand_list = get_car_prop(car, 'car_brand')
    if brand_list:
        b = brand_list[0]
        if b in ('ford', 'tesla'):
            countries.append('american')
        if b in ('toyota', 'honda'):
            countries.append('japanese')
    return countries


def matches(car, key, val):
    """Проверка: соответствует ли автомобиль одному критерию."""
    if key == 'brand':
        return val in get_car_prop(car, 'car_brand')
    if key == 'body_type':
        return val in get_car_prop(car, 'car_body_type')
    if key == 'fuel':
        return val in get_car_prop(car, 'car_fuel')
    if key == 'engine':
        return val in get_car_prop(car, 'car_engine')
    if key == 'transmission':
        return val in get_car_prop(car, 'car_transmission')
    if key == 'country':
        return val in get_country(car)

    powers = get_car_prop(car, 'car_power')
    if not powers:
        return False
    p = int(powers[0])
    if key == 'min_power':
        return p >= val
    if key == 'max_power':
        return p <= val
    return False


def count_matches(car, prefs):
    return sum(1 for k, v in prefs if matches(car, k, v))

def find_cars(prefs):
    full, partial = [], []
    for car in get_all_cars():
        score = count_matches(car, prefs)
        if score == len(prefs):
            full.append(car)
        elif score > 0:
            partial.append((car, score))

    partial.sort(key=lambda x: x[1], reverse=True)
    return full, partial

def display_car(car, score=None):
    brand_list = get_car_prop(car, 'car_brand')
    body_list  = get_car_prop(car, 'car_body_type')
    fuel_list  = get_car_prop(car, 'car_fuel')
    eng_list   = get_car_prop(car, 'car_engine')
    trans_list = get_car_prop(car, 'car_transmission')
    power_list = get_car_prop(car, 'car_power')

    brand = brand_list[0] if brand_list else '?'
    body  = body_list[0]  if body_list  else '?'
    fuel  = fuel_list[0]  if fuel_list  else '?'
    eng   = eng_list[0]   if eng_list   else '?'
    trans = trans_list[0] if trans_list else '?'
    power = power_list[0] if power_list else '?'

    if score is not None:
        print(f"{car}  [совпадений: {score}]")
    else:
        print(f"{car}")
    print(f"Марка: {brand}, Кузов: {body}, Топливо: {fuel}, Двигатель: {eng}, КПП: {trans}, Мощность: {power} л.с.")
    print()


def describe_prefs(prefs):
    if not prefs:
        return
    print("Распознанные критерии:")
    for key, val in prefs:
        print(f"{key} = {val}")
    print()


def show_recommendations(prefs):
    print()
    print("Результаты подбора")
    full, partial = find_cars(prefs)

    if full:
        print("Найдены автомобили, полностью соответствующие запросу:")
        print()
        for car in full:
            display_car(car)
    else:
        print("Нет автомобилей, полностью соответствующих всем критериям.")
        print()
        if partial:
            print("Ближайшие варианты (по количеству совпадений):")
            print()
            for car, score in partial[:3]:
                display_car(car, score)
        else:
            print("К сожалению, ничего не найдено.")


def print_greeting():
    print()
    print("Система подбора автомобиля")
    print()
    print("Опишите, какой автомобиль вы хотите, обычными словами.")
    print()
    print("Можно упоминать:")
    print("марку: bmw, audi, tesla, тойота, мерседес, ...")
    print("кузов: седан, купе, внедорожник, спорткар, ...")
    print("топливо: бензиновый, дизельный, гибрид, электромобиль, ...")
    print("коробку: автомат, механика, вариатор, робот, ...")
    print("страну: немецкую, японскую, американскую, ...")
    print("мощность: 'от 400 лошадей', 'до 300 л.с.', ...")
    print()
    print("Пример: Хочу немецкую машину, бензиновую, мощностью от 400")
    print()


def dialog_loop():
    while True:
        try:
            user_input = input("Ваш запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if user_input.lower() == 'exit':
            print("До свидания!")
            break
        if not user_input:
            print("Пустой ввод. Попробуйте снова.")
            continue

        prefs = parse_input(user_input)

        if not prefs:
            print("Не удалось распознать ни одного критерия.")
            print("Попробуйте переформулировать запрос.")
            continue

        describe_prefs(prefs)
        show_recommendations(prefs)

        print()
        try:
            ans = input("Хотите попробовать снова? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if ans != 'y':
            print("До свидания!")
            break
        print()


def main():
    try:
        list(prolog.query("X = 1"))
    except Exception as e:
        print(f"Ошибка с прологом: {e}")
        sys.exit(1)

    if not load_knowledge_base("cars.pl"):
        sys.exit(1)

    print_greeting()
    dialog_loop()


if __name__ == "__main__":
    main()