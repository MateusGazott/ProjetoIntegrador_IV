"""
Testes Unitários do Módulo da Groq (Integrante 3)
Verifica validações, estrutura de prompts e integração com mock.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import groq_service

class TestGroqService(unittest.TestCase):

    def test_validacao_mensagem_vazia(self):
        with self.assertRaises(ValueError):
            groq_service.validar_entradas("", "Resumo teste")

    def test_validacao_resumo_vazio(self):
        with self.assertRaises(ValueError):
            groq_service.validar_entradas("Como agir?", "")

    def test_montagem_mensagens(self):
        resumo = "[DIAGNÓSTICO] Alto Risco - 65% de chance de insolvência"
        mensagens = groq_service.montar_mensagens(
            mensagem_usuario="O que devo fazer primeiro?",
            resumo_diagnostico=resumo,
            historico=[{"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Olá, como posso ajudar?"}]
        )

        self.assertEqual(mensagens[0]["role"], "system")
        self.assertIn("FinGuard AI", mensagens[0]["content"])
        self.assertIn("Alto Risco", mensagens[1]["content"])
        self.assertEqual(mensagens[2]["role"], "user")
        self.assertEqual(mensagens[3]["role"], "assistant")
        self.assertEqual(mensagens[4]["content"], "O que devo fazer primeiro?")

    def test_falta_chave_api_lanca_excecao(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                groq_service.obter_cliente_groq()
            self.assertIn("GROQ_API_KEY", str(ctx.exception))

    @patch("groq.Groq")
    def test_gerar_orientacao_sucesso(self, mock_groq_class):
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Recomenda-se negociar prazos com fornecedores e reter lucros."
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test123456"}):
            resposta = groq_service.gerar_orientacao(
                "Quais os primeiros passos?",
                "[DIAGNÓSTICO] Empresa em Alto Risco",
                []
            )
            self.assertIn("fornecedores", resposta)

if __name__ == "__main__":
    unittest.main()
