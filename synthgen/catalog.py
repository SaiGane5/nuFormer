"""Merchant catalog: category, amount distribution, price tier and statement-style description templates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    SALARY = "SALARY"; RENT = "RENT"; UTILITIES = "UTILITIES"; SUBSCRIPTION = "SUBSCRIPTION"
    GROCERY = "GROCERY"; RESTAURANT = "RESTAURANT"; DELIVERY = "DELIVERY"; TRANSPORT = "TRANSPORT"
    FUEL = "FUEL"; SHOPPING = "SHOPPING"; PHARMACY = "PHARMACY"; BABY = "BABY"; AUTO = "AUTO"
    TRAVEL_AIR = "TRAVEL_AIR"; TRAVEL_HOTEL = "TRAVEL_HOTEL"; TRAVEL_ABROAD = "TRAVEL_ABROAD"
    INVEST = "INVEST"; INVEST_INCOME = "INVEST_INCOME"; PIX_OUT = "PIX_OUT"; PIX_IN = "PIX_IN"
    FEE = "FEE"; LOAN_IN = "LOAN_IN"; COLLECTION = "COLLECTION"; CASH = "CASH"


@dataclass(frozen=True)
class Merchant:
    name: str
    category: Category
    mu: float          # log-amount location at median income
    sigma: float       # log-amount scale
    tier: int          # -1 budget, 0 mid, +1 premium (income-dependent choice)
    templates: tuple
    inflow: bool = False


C = Category
_M = [
    # ---- inflows
    Merchant("SALARY", C.SALARY, 0, 0, 0, ("SALARIO {company}", "PAGTO SALARIO {company}", "CRED SALARIO {company} LTDA"), True),
    Merchant("PIX_IN", C.PIX_IN, 4.6, 1.0, 0, ("PIX RECEBIDO {name}", "TRANSF RECEBIDA {name}"), True),
    Merchant("RENDIMENTO", C.INVEST_INCOME, 3.0, 0.8, 0, ("RENDIMENTO CDB", "RENDIMENTO CAIXINHA", "JUROS POUPANCA"), True),
    Merchant("CREDITAS RAPIDO", C.LOAN_IN, 7.0, 0.5, 0, ("EMPRESTIMO PESSOAL CREDITAS", "CRED RAPIDO FINANCEIRA {num}", "ANTECIPACAO SAQUE FGTS"), True),
    # ---- recurring bills
    Merchant("ALUGUEL", C.RENT, 7.3, 0.25, 0, ("ALUGUEL {name}", "PIX ALUGUEL IMOB {city}", "QUINTOANDAR ALUGUEL"),),
    Merchant("ENEL", C.UTILITIES, 5.0, 0.3, 0, ("ENEL DISTRIB {city}", "CONTA LUZ ENEL"),),
    Merchant("SABESP", C.UTILITIES, 4.4, 0.3, 0, ("SABESP AGUA {city}",),),
    Merchant("VIVO", C.UTILITIES, 4.7, 0.2, 0, ("VIVO FIBRA", "TELEFONICA VIVO {num}"),),
    Merchant("CLARO", C.UTILITIES, 4.6, 0.2, 0, ("CLARO NET VIRTUA", "CLARO CELULAR"),),
    # ---- subscriptions
    Merchant("NETFLIX", C.SUBSCRIPTION, 3.7, 0.05, 0, ("NETFLIX.COM", "DL*NETFLIX"),),
    Merchant("SPOTIFY", C.SUBSCRIPTION, 3.0, 0.05, 0, ("SPOTIFY P{num}", "DL*SPOTIFY"),),
    Merchant("AMAZON PRIME", C.SUBSCRIPTION, 2.7, 0.05, 0, ("AMAZON PRIME BR", "AMZN PRIME*{num}"),),
    Merchant("SMARTFIT", C.SUBSCRIPTION, 4.6, 0.1, 0, ("SMART FIT {city}", "SMARTFIT MENSALIDADE"),),
    Merchant("CHATGPT", C.SUBSCRIPTION, 4.6, 0.05, 1, ("OPENAI *CHATGPT SUBSCR",),),
    Merchant("DISNEY", C.SUBSCRIPTION, 3.4, 0.05, 0, ("DISNEY PLUS",),),
    # ---- groceries
    Merchant("ASSAI", C.GROCERY, 4.6, 0.7, -1, ("ASSAI ATACADISTA {num}", "ASSAI {city}"),),
    Merchant("ATACADAO", C.GROCERY, 4.7, 0.7, -1, ("ATACADAO {num} {city}",),),
    Merchant("CARREFOUR", C.GROCERY, 4.5, 0.7, 0, ("CARREFOUR {city}", "CARREFOUR EXP {num}"),),
    Merchant("PAO DE ACUCAR", C.GROCERY, 4.6, 0.7, 1, ("PAO DE ACUCAR {num}", "GPA PDA {city}"),),
    Merchant("ST MARCHE", C.GROCERY, 4.8, 0.6, 1, ("ST MARCHE {city}",),),
    Merchant("MERCADINHO", C.GROCERY, 3.4, 0.7, -1, ("MERCADINHO {name}", "MERCEARIA {name}"),),
    # ---- restaurants / food
    Merchant("PADARIA", C.RESTAURANT, 2.9, 0.6, -1, ("PADARIA {name}", "PANIFICADORA {city}"),),
    Merchant("MCDONALDS", C.RESTAURANT, 3.6, 0.4, -1, ("MCDONALDS {num}", "ARCOS DOURADOS {city}"),),
    Merchant("OUTBACK", C.RESTAURANT, 5.0, 0.4, 1, ("OUTBACK STEAKHOUSE {city}",),),
    Merchant("RESTAURANTE", C.RESTAURANT, 4.1, 0.6, 0, ("RESTAURANTE {name}", "REST {name} {city}"),),
    Merchant("STARBUCKS", C.RESTAURANT, 3.4, 0.3, 1, ("STARBUCKS {city}",),),
    Merchant("BAR", C.RESTAURANT, 4.2, 0.7, 0, ("BAR DO {name}", "BOTECO {name}"),),
    Merchant("IFOOD", C.DELIVERY, 3.9, 0.5, 0, ("IFOOD *{name}", "IFD*IFOOD.COM {num}"),),
    Merchant("RAPPI", C.DELIVERY, 4.0, 0.5, 0, ("RAPPI*{name}", "RAPPI BRASIL"),),
    Merchant("ZE DELIVERY", C.DELIVERY, 4.2, 0.4, 0, ("ZE DELIVERY BEBIDAS",),),
    # ---- transport / auto
    Merchant("UBER", C.TRANSPORT, 3.1, 0.6, 0, ("UBER *TRIP {city}", "UBER* PENDING", "UBER DO BRASIL"),),
    Merchant("99", C.TRANSPORT, 2.9, 0.6, -1, ("99APP *99POP", "99 TECNOLOGIA {city}"),),
    Merchant("METRO", C.TRANSPORT, 1.6, 0.3, -1, ("BILHETE UNICO {city}", "RECARGA METRO SP"),),
    Merchant("SEM PARAR", C.TRANSPORT, 3.5, 0.5, 1, ("SEM PARAR TAG", "VELOE PEDAGIO"),),
    Merchant("SHELL", C.FUEL, 5.1, 0.4, 0, ("SHELL POSTO {num}", "RAIZEN SHELL {city}"),),
    Merchant("IPIRANGA", C.FUEL, 5.1, 0.4, 0, ("POSTO IPIRANGA {city}",),),
    Merchant("CONCESSIONARIA", C.AUTO, 9.3, 0.5, 0, ("CONCESSIONARIA {brand} {city}", "ENTRADA VEICULO {brand}"),),
    Merchant("AUTOPECAS", C.AUTO, 5.0, 0.7, 0, ("AUTO PECAS {name}", "PNEUSTORE", "DETRAN IPVA LICENC"),),
    # ---- shopping
    Merchant("MERCADO LIVRE", C.SHOPPING, 4.4, 0.9, 0, ("MERCADOLIVRE*{name}", "MP*MERCADOLIVRE"),),
    Merchant("AMAZON", C.SHOPPING, 4.5, 0.9, 0, ("AMAZON BR", "AMZN MKTP BR {num}"),),
    Merchant("SHEIN", C.SHOPPING, 4.3, 0.6, -1, ("SHEIN.COM", "SHEIN *{num}"),),
    Merchant("RENNER", C.SHOPPING, 4.8, 0.6, 0, ("LOJAS RENNER {city}",),),
    Merchant("MAGALU", C.SHOPPING, 5.6, 1.0, 0, ("MAGAZINE LUIZA {num}", "MAGALU *PARC {num}"),),
    Merchant("APPLE", C.SHOPPING, 5.5, 1.2, 1, ("APPLE.COM/BILL", "APPLE STORE {city}"),),
    Merchant("ZARA", C.SHOPPING, 5.4, 0.5, 1, ("ZARA BRASIL {city}",),),
    # ---- health / baby
    Merchant("DROGASIL", C.PHARMACY, 4.0, 0.7, 0, ("DROGASIL {num}", "RAIA DROGASIL {city}"),),
    Merchant("PAGUE MENOS", C.PHARMACY, 3.8, 0.7, -1, ("PAGUE MENOS {num}",),),
    Merchant("LABORATORIO", C.PHARMACY, 5.0, 0.6, 1, ("FLEURY MEDICINA", "DASA LAB {city}", "CONSULTA DR {name}"),),
    Merchant("BABY", C.BABY, 5.0, 0.7, 0, ("PBKIDS {city}", "CENTRAL DO BEBE", "FRALDAS {name}", "MATERNIDADE {city}"),),
    # ---- travel
    Merchant("LATAM", C.TRAVEL_AIR, 7.1, 0.5, 1, ("LATAM AIRLINES {num}", "TAM LINHAS AEREAS"),),
    Merchant("GOL", C.TRAVEL_AIR, 6.7, 0.5, 0, ("GOL LINHAS AEREAS {num}", "GOL TRANSP AEREOS"),),
    Merchant("AZUL", C.TRAVEL_AIR, 6.8, 0.5, 0, ("AZUL LINHAS AEREAS",),),
    Merchant("BOOKING", C.TRAVEL_HOTEL, 6.6, 0.6, 0, ("BOOKING.COM HOTEL {city}", "HOTEL AT BOOKING.COM"),),
    Merchant("AIRBNB", C.TRAVEL_HOTEL, 6.7, 0.6, 0, ("AIRBNB * HM{num}",),),
    Merchant("DECOLAR", C.TRAVEL_HOTEL, 7.0, 0.5, 0, ("DECOLAR.COM PACOTE", "123MILHAS {num}"),),
    Merchant("ABROAD", C.TRAVEL_ABROAD, 4.4, 0.8, 1, ("{intl} {country}", "IOF COMPRA INTERNACIONAL", "WISE FX {country}"),),
    # ---- investing / transfers / cash
    Merchant("BROKER", C.INVEST, 6.8, 0.7, 1, ("APLICACAO CDB NU", "XP INVESTIMENTOS TED", "GUARDAR CAIXINHA", "TESOURO DIRETO APLIC"),),
    Merchant("PIX_OUT", C.PIX_OUT, 4.3, 1.0, 0, ("PIX ENVIADO {name}", "TRANSF ENVIADA {name}"),),
    Merchant("SAQUE", C.CASH, 4.8, 0.5, -1, ("SAQUE BANCO24H {city}", "SAQUE ATM {num}"),),
    # ---- stress
    Merchant("FEES", C.FEE, 3.2, 0.5, 0, ("TARIFA CHEQUE ESPECIAL", "JUROS ROTATIVO CARTAO", "MULTA ATRASO FATURA", "IOF CHEQUE ESPECIAL"),),
    Merchant("COLLECTION", C.COLLECTION, 5.3, 0.6, 0, ("SERASA LIMPA NOME ACORDO", "RECOVERY COBRANCA {num}", "ACORDO DIVIDA {company}"),),
]
CATALOG: tuple = tuple(_M)

FILL = {
    "city": ["SAO PAULO", "SP", "RIO DE JANEIRO", "RJ", "BELO HORIZONTE", "CURITIBA", "PORTO ALEGRE", "RECIFE",
             "SALVADOR", "FORTALEZA", "BRASILIA", "CAMPINAS", "GOIANIA", "MANAUS"],
    "name": ["JOAO S", "MARIA O", "ANA P", "CARLOS M", "LUCAS F", "JULIANA R", "PEDRO H", "BEATRIZ L", "RAFAEL C",
             "FERNANDA A", "SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "COSTA", "LIMA", "PEREIRA", "ALMEIDA"],
    "company": ["ACME", "TOTVS", "AMBEV", "ITAU UNIBANCO", "MAGAZINE LUIZA", "PREFEITURA MUN", "HOSPITAL SANTA CASA",
                "CONSTRUTORA ALFA", "CNPJ 12345678", "INFRATEC", "LOGIST BRASIL"],
    "brand": ["FIAT", "VW", "TOYOTA", "HYUNDAI", "CHEVROLET", "HONDA", "RENAULT"],
    "intl": ["WALMART", "STARBUCKS", "UBER BV", "ZARA", "CARREFOUR FR", "TARGET", "MCDONALDS", "DUTY FREE"],
    "country": ["USA", "PRT", "ARG", "CHL", "FRA", "ITA", "ESP", "MEX"],
}

BY_CATEGORY: dict = {}
for _i, _m in enumerate(CATALOG):
    BY_CATEGORY.setdefault(_m.category, []).append(_i)

# coarse buckets exposed to the tabular model (lossy on purpose — LLD 2.1 features.py)
COARSE = {
    C.GROCERY: "grocery", C.RESTAURANT: "food", C.DELIVERY: "food", C.TRANSPORT: "transport", C.FUEL: "transport",
    C.AUTO: "other", C.SHOPPING: "shopping", C.PHARMACY: "health", C.BABY: "shopping",
    C.TRAVEL_AIR: "other", C.TRAVEL_HOTEL: "other", C.TRAVEL_ABROAD: "shopping",
    C.RENT: "bills", C.UTILITIES: "bills", C.SUBSCRIPTION: "bills", C.FEE: "bills", C.COLLECTION: "bills",
    C.INVEST: "transfers", C.PIX_OUT: "transfers", C.CASH: "other",
}
