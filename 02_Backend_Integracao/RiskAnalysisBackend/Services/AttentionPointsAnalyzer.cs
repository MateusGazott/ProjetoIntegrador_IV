using System.Globalization;

namespace RiskAnalysisBackend.Services;

public interface IAttentionPointsAnalyzer
{
    List<string> Analisar(IReadOnlyDictionary<string, float> indicadores, IReadOnlyCollection<string> imputados);
}

public sealed class AttentionPointsAnalyzer : IAttentionPointsAnalyzer
{
    private sealed record Regra(string Atributo, Func<float, bool> Condicao, Func<string, string> Mensagem);

    private static readonly Regra[] Regras =
    {
        new("Attr1", v => v < 0f, v => $"Rentabilidade negativa: lucro líquido sobre ativo total em {v}."),
        new("Attr2", v => v > 0.7f, v => $"Endividamento elevado: passivo total sobre ativo total em {v}."),
        new("Attr3", v => v < 0f, v => $"Capital de giro negativo ({v} do ativo total)."),
        new("Attr4", v => v < 1f, v => $"Liquidez corrente abaixo de 1 ({v}): o ativo circulante não cobre o passivo circulante."),
        new("Attr6", v => v < 0f, v => $"Lucros retidos negativos ({v}): indício de prejuízos acumulados."),
        new("Attr7", v => v < 0f, v => $"EBIT negativo ({v}): a operação não cobre os próprios custos."),
        new("Attr8", v => v < 0.5f, v => $"Patrimônio líquido baixo frente ao passivo total ({v})."),
        new("Attr10", v => v < 0.2f, v => $"Patrimônio líquido representa pouco do ativo total ({v}).")
    };

    public List<string> Analisar(IReadOnlyDictionary<string, float> indicadores, IReadOnlyCollection<string> imputados)
    {
        var pontos = new List<string>();

        foreach (var regra in Regras)
        {
            if (imputados.Contains(regra.Atributo, StringComparer.OrdinalIgnoreCase)) continue;
            if (!indicadores.TryGetValue(regra.Atributo, out float valor)) continue;
            if (!regra.Condicao(valor)) continue;

            pontos.Add(regra.Mensagem(valor.ToString("0.####", CultureInfo.InvariantCulture)));
        }

        return pontos;
    }
}
