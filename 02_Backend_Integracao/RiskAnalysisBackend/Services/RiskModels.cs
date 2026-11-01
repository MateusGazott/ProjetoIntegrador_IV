using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using RiskAnalysisBackend.Configuration;
using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Services;

public interface IRiskModel
{
    string Modo { get; }
    bool EhModeloReal { get; }
    PredictionResult Prever(float[] vetor, IReadOnlyDictionary<string, float> indicadores);
}

public sealed class OnnxRiskModel : IRiskModel, IDisposable
{
    private readonly InferenceSession _sessao;
    private readonly string _nomeEntrada;
    private readonly int _totalAtributos;

    public string Modo => "ONNX";
    public bool EhModeloReal => true;

    public OnnxRiskModel(string caminhoModelo, ModelMetadata metadata)
    {
        _totalAtributos = metadata.FeatureNames.Count;
        _sessao = new InferenceSession(caminhoModelo);
        _nomeEntrada = _sessao.InputMetadata.Keys.First();

        int[] dimensoes = _sessao.InputMetadata[_nomeEntrada].Dimensions;
        if (dimensoes.Length >= 2 && dimensoes[^1] > 0 && dimensoes[^1] != _totalAtributos)
        {
            int esperado = dimensoes[^1];
            _sessao.Dispose();
            throw new InvalidOperationException(
                $"O modelo ONNX espera {esperado} atributos, mas model_metadata.json define {_totalAtributos}. Reexporte os dois arquivos juntos.");
        }
    }

    public PredictionResult Prever(float[] vetor, IReadOnlyDictionary<string, float> indicadores)
    {
        if (vetor.Length != _totalAtributos)
            throw new InvalidOperationException("O vetor de entrada não corresponde ao número de atributos do modelo.");

        var tensor = new DenseTensor<float>(vetor, new[] { 1, _totalAtributos });
        var entradas = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_nomeEntrada, tensor) };

        using var resultados = _sessao.Run(entradas);

        float[]? probabilidades = ExtrairProbabilidades(resultados);
        if (probabilidades != null)
        {
            float pSaudavel = Math.Clamp(probabilidades[0], 0f, 1f);
            float pInsolvencia = Math.Clamp(probabilidades[1], 0f, 1f);
            float soma = pSaudavel + pInsolvencia;

            if (soma <= 0f)
                throw new InvalidOperationException("O modelo retornou probabilidades inválidas.");

            if (Math.Abs(soma - 1f) > 0.01f)
            {
                pSaudavel /= soma;
                pInsolvencia /= soma;
            }

            return new PredictionResult(pInsolvencia >= 0.5f ? 1 : 0, pSaudavel, pInsolvencia);
        }

        int classe = resultados.First().AsEnumerable<long>().First() == 1 ? 1 : 0;
        return new PredictionResult(classe, 1f - classe, classe);
    }

    private static float[]? ExtrairProbabilidades(IEnumerable<DisposableNamedOnnxValue> resultados)
    {
        foreach (var saida in resultados)
        {
            try
            {
                float[]? valores = saida.AsTensor<float>()?.ToArray();
                if (valores is { Length: 2 } && valores.All(float.IsFinite))
                    return valores;
            }
            catch (Exception)
            {
            }
        }
        return null;
    }

    public void Dispose() => _sessao.Dispose();
}

public sealed class FallbackRiskModel : IRiskModel
{
    private readonly IReadOnlyDictionary<string, float> _medianas;

    public string Modo => "FallbackHeuristico";
    public bool EhModeloReal => false;

    public FallbackRiskModel(ModelMetadata metadata) => _medianas = metadata.Medians;

    public PredictionResult Prever(float[] vetor, IReadOnlyDictionary<string, float> indicadores)
    {
        float roa = Valor(indicadores, "Attr1");
        float endividamento = Valor(indicadores, "Attr2");
        float capitalGiro = Valor(indicadores, "Attr3");
        float liquidez = Valor(indicadores, "Attr4");

        float score = 0.30f;
        score += (endividamento - 0.50f) * 0.65f;
        score -= roa * 0.80f;
        score -= capitalGiro * 0.40f;
        if (liquidez < 1.0f) score += 0.20f;

        float pInsolvencia = Math.Clamp(score, 0.02f, 0.98f);
        return new PredictionResult(pInsolvencia >= 0.5f ? 1 : 0, 1f - pInsolvencia, pInsolvencia);
    }

    private float Valor(IReadOnlyDictionary<string, float> indicadores, string nome)
    {
        if (indicadores.TryGetValue(nome, out float valor)) return valor;
        return _medianas.TryGetValue(nome, out float mediana) ? mediana : 0f;
    }
}

public static class RiskModelFactory
{
    public static IRiskModel Criar(MLOptions options, ModelMetadata metadata, ILogger logger)
    {
        string caminho = MLOptions.ResolverCaminho(options.ModelPath);

        if (!File.Exists(caminho))
        {
            logger.LogWarning("Modelo ONNX não encontrado em {Caminho}. Operando em modo heurístico provisório.", caminho);
            return new FallbackRiskModel(metadata);
        }

        var modelo = new OnnxRiskModel(caminho, metadata);
        logger.LogInformation("Modelo ONNX carregado de {Caminho} com {Total} atributos.", caminho, metadata.FeatureNames.Count);
        return modelo;
    }
}
