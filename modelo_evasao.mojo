from python import Python

fn main() raises:
    var py = Python.import_module('builtins')
    var joblib = Python.import_module('joblib')
    var pd = Python.import_module('pandas')

    var modelo = joblib.load('modelo_evasao.pkl')

    classificacaoRacial = py.input('Classificação Racial: ')
    sexo = py.input('Sexo: ')
    fonteFinanciamento = py.input('Fonte de Financiamento (Digite 0 para Sem Programa Associado e qualquer valor para Recursos Orçamentários): ')
    rendaFamiliarNum = py.input('Renda Familiar (Numérica): ')
    faixaEtariaNum = py.input('Faixa Etária (Numérica): ')

    novosDados = pd.DataFrame({
        'ClassificacaoRacial': [classificacaoRacial],
        'Sexo': [sexo],
        'FonteFinanciamento': ['Sem Programa Associado' if fonteFinanciamento == 0 else 'Recursos Orçamentários'],
        'RendaFamiliarNum': [rendaFamiliarNum],
        'FaixaEtariaNum': [faixaEtariaNum]
    })

    probabilidade = modelo.predict_proba(novosDados)

    print('Probabilidade de evasão:', probabilidade[0][1])
