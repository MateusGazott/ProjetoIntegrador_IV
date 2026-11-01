using System.Globalization;
using Microsoft.Extensions.Options;
using RiskAnalysisBackend.Configuration;
using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Services;

public interface IFeaturePreprocessor
{
    PreprocessingResult Processar(Dictionary<string, double?> indicadoresEntrada);
}

public sealed class FeaturePreprocessor : IFeaturePreprocessor
{
    private const int MaximoIndicadoresNaRequisicao = 200;

    private readonly ModelMetadata _metadata;
    private readonly float _limiteExtremo;

    public FeaturePreprocessor(ModelMetadata metadata, IOptions<MLOptions> options)
    {
        _metadata = metadata;
        _limiteExtremo = options.Value.LimiteValorExtremo;
    }

    public PreprocessingResult Processar(Dictionary<string, double?> indicadoresEntrada)
    {
        if (indicadoresEntrada.Count > MaximoIndicadoresNaRequisicao)
            throw new ArgumentException($"Foram enviados indicadores demais (máximo de {MaximoIndicadoresNaRequisicao}).");

        var entrada = new Dictionary<string, double?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (chave, valor) in indicadoresEntrada)
        {
            string nome = (chave ?? string.Empty).Trim();
            if (nome.Length == 0)
                throw new ArgumentException("Há um indicador sem nome na requisição.");
            if (!entrada.TryAdd(nome, valor))
                throw new ArgumentException($"O indicador '{nome}' foi informado mais de uma vez.");
        }

        int total = _metadata.FeatureNames.Count;
        var vetor = new float[total];
        var processados = new Dictionary<string, float>(total);
        var imputados = new List<string>();
        var alertas = new List<string>();

        for (int i = 0; i < total; i++)
        {
            string nome = _metadata.FeatureNames[i];
            float valorFinal;

            if (entrada.TryGetValue(nome, out double? bruto) && bruto.HasValue && double.IsFinite(bruto.Value))
            {
                valorFinal = (float)bruto.Value;

                if (!float.IsFinite(valorFinal))
                    throw new ArgumentException($"O valor de '{nome}' está fora do intervalo numérico suportado.");

                if (Math.Abs(valorFinal) > _limiteExtremo)
                    alertas.Add($"O indicador '{nome}' apresenta valor extremo ({valorFinal.ToString("G6", CultureInfo.InvariantCulture)}). Verifique a escala informada.");
            }
            else
            {
                valorFinal = _metadata.Medians[nome];
                imputados.Add(nome);
            }

            vetor[i] = valorFinal;
            processados[nome] = valorFinal;
        }

        if (imputados.Count == total)
            throw new ArgumentException(
                $"Nenhum dos indicadores enviados corresponde aos atributos esperados pelo modelo ({string.Join(", ", _metadata.FeatureNames)}).");

        var conhecidos = new HashSet<string>(_metadata.FeatureNames, StringComparer.OrdinalIgnoreCase);
        foreach (string nome in entrada.Keys.Where(k => !conhecidos.Contains(k)))
            alertas.Add($"O indicador '{nome}' não é utilizado pelo modelo e foi ignorado.");

        return new PreprocessingResult(vetor, processados, imputados, alertas);
    }
}
