import gc
import importlib.util
import io
import json
import os
from contextlib import contextmanager, redirect_stderr
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from activations import select_positions, find_answer_token_index
from config import RunConfig, parse_config
from llm import generate, generate_samples, load_model


class SelectionTests(unittest.TestCase):
    def test_answer_anchor_and_relative_positions(self):
        class Tokenizer:
            def decode(self, ids):
                return {0: 'Yes', 1: ' no ', 2: ' YES\n', 3: 'Answer:',
                        4: 'nobody', 5: '<eos>', 6: 'Yes.'}[ids[0]]
        tokenizer = Tokenizer()
        ids = [0, 3, 1, 4, 2, 5]
        anchor = find_answer_token_index(tokenizer, ids, 2)
        self.assertEqual(anchor, 4)
        self.assertIsNone(find_answer_token_index(tokenizer, [0, 3, 4, 6, 5], 1))
        self.assertIsNone(find_answer_token_index(tokenizer, [0], 1))
        self.assertEqual(select_positions('answer-tokens-1,answer-tokens,answer-tokens+1,answer-tokens',
                         2, 6, answer_token_index=anchor), [3, 4, 5])
        self.assertEqual(select_positions('answer-tokens-1', 2, 3, answer_token_index=2), [1])
        self.assertEqual(select_positions('answer-tokens', 2, 6), [])
        for spec in ('answer-tokens-5', 'answer-tokens+2'):
            with self.assertRaisesRegex(ValueError, 'outside sequence'):
                select_positions(spec, 2, 6, answer_token_index=anchor)
        for spec in ('answer-tokens', 'answer-tokens-1,answer-tokens+1'):
            cfg, _ = parse_config(['--backend', 'transformers', '--model', 'org/model',
                                   '--activation-positions', spec])
            self.assertEqual(cfg.activation_positions, spec)
        for spec in ('answer-tokens+', 'answer-tokens,0', 'answer-tokens--1', 'answer-tokens,'):
            with self.assertRaises(ValueError):
                RunConfig('org/model', Path('questions.json'), backend='transformers', activation_positions=spec)

    def test_positions(self):
        for spec, expected in [('last-prompt-token', [2]), ('last-token', [4]),
                               ('prompt', [0, 1, 2]), ('response', [3, 4]),
                               ('all', [0, 1, 2, 3, 4]), ('0,-1', [0, 4])]:
            with self.subTest(spec=spec):
                self.assertEqual(select_positions(spec, 3, 5), expected)
        self.assertEqual(select_positions('response', 3, 3), [])
        with self.assertRaises(ValueError):
            select_positions('5', 3, 5)

    def test_flags_and_validation(self):
        config, _ = parse_config(['--backend', 'transformers', '--model', 'org/model',
                                  '--extract-activations', '--activation-layers', '0,2'])
        self.assertTrue(config.extract_activations)
        self.assertEqual(config.metadata('integrity', 1)['model'], 'org/model')
        for changes in [dict(backend='llama-cpp'), dict(activation_layers='-1'),
                        dict(activation_layers=''), dict(activation_types='unknown'),
                        dict(activation_positions='nonsense')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.gguf'
            model.touch()
            cfg, _ = parse_config(['--backend', 'gguf', '--model', str(model)])
            self.assertEqual(cfg.backend, 'llama-cpp')
            self.assertFalse(cfg.extract_activations)

    def test_runpod_rejects_local_execution(self):
        cfg = RunConfig('org/model', Path('questions.json'), backend='transformers', runtime='runpod')
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'inside'):
            with load_model(cfg):
                self.fail('Should fail before loading')


@unittest.skipUnless(importlib.util.find_spec('transformer_lens'), 'install requirements-activations.txt')
class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.addClassCleanup(gc.collect)
        cls.path = Path(cls.directory.name)
        raw = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, '<pad>': 2, 'system': 3,
            'user': 4, 'assistant': 5, 'Yes': 6, 'No': 7, 'Answer': 8, ':': 9,
            'Confidence': 10, '80': 11, '%': 12}, unk_token='<unk>'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='<unk>',
                                            eos_token='<eos>', pad_token='<pad>')
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(cls.path)
        with torch.random.fork_rng():
            torch.manual_seed(2)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=13, n_positions=512, n_embd=16,
                n_layer=2, n_head=2, bos_token_id=1, eos_token_id=1, pad_token_id=2))
            model.save_pretrained(cls.path)

    def config(self, **kwargs):
        return RunConfig(str(self.path), self.path / 'questions.json', backend='transformers',
                         device='cpu', n_ctx=512, max_tokens=2, **kwargs)

    def test_real_capture_and_forward_parity(self):
        import torch
        from activations import ActivationRecorder
        from safetensors.torch import load_file
        with load_model(self.config()) as llm:
            tokens = torch.tensor([[3, 6, 4, 7, 5]])
            with torch.inference_mode():
                before = llm.model(tokens).logits.clone()
            recorder = ActivationRecorder(llm.model, llm.tokenizer,
                self.config(extract_activations=True, activation_layers='0,1',
                            activation_types='resid_pre,resid_post,attn_out,mlp_out'))
            llm.activation_recorder = recorder
            with torch.inference_mode():
                after = recorder.bridge(tokens)
            torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
            with tempfile.TemporaryDirectory() as out:
                llm.begin_run(out)
                response = generate(llm, 'Yes', 'No', seed=3, temperature=0.7, max_tokens=2,
                                    activation_context={'id': 'q1', 'condition': 'control', 'sample': 0})
                self.assertIsInstance(response, str)
                directory = Path(out) / 'activations'
                record = json.loads((directory / 'index.jsonl').read_text())
                self.assertEqual(record['context']['id'], 'q1')
                self.assertEqual(record['positions'], [record['prompt_length'] - 1])
                cache = load_file(str(directory / record['file']))
                self.assertEqual(len(cache), 8)
                for value in cache.values():
                    self.assertEqual(tuple(value.shape), (1, 1, 16))
                with torch.inference_mode():
                    _, full = recorder.bridge.run_with_cache(torch.tensor([record['token_ids']]),
                                                              names_filter=recorder.names)
                for name, value in cache.items():
                    torch.testing.assert_close(value, full[name][:, record['positions'], :], rtol=1e-4, atol=1e-5)
                del full, cache, value

    def test_qwen3_capture_and_parity(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
        from activations import ActivationRecorder
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=13, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4, max_position_embeddings=512,
            attn_implementation='eager'))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        tokens = torch.tensor([[3, 6, 4, 7, 5]])
        with torch.inference_mode():
            before = model(tokens).logits.clone()
        recorder = ActivationRecorder(model, tokenizer, self.config(extract_activations=True,
            activation_layers='all', activation_types='resid_pre,resid_post,attn_out,mlp_out'))
        with torch.inference_mode():
            after = recorder.bridge(tokens)
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
        with tempfile.TemporaryDirectory() as out:
            recorder.begin_run(out)
            recorder.capture(tokens, 3, context={'id': 'qwen'}, seed=None, temperature=0)
            record = json.loads((Path(out) / 'activations/index.jsonl').read_text())
            self.assertEqual(len(record['hooks']), 8)
            self.assertEqual(record['positions'], [2])

    def test_disabled_does_not_import_lens_or_write(self):
        with patch.dict('sys.modules', {'transformer_lens': None}):
            with load_model(self.config()) as llm:
                self.assertIsNone(llm.activation_recorder)
                generate(llm, 'Yes', 'No', max_tokens=2)

    def test_response_positions_and_seed_preservation(self):
        import torch
        with load_model(self.config(extract_activations=True, activation_positions='response')) as llm:
            with tempfile.TemporaryDirectory() as out:
                llm.begin_run(out)
                state = torch.random.get_rng_state().clone()
                a = generate(llm, 'Yes', 'No', temperature=0.7, seed=12, max_tokens=2)
                b = generate(llm, 'Yes', 'No', temperature=0.7, seed=12, max_tokens=2)
                self.assertEqual(a, b)
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                records = [json.loads(line) for line in (Path(out) / 'activations/index.jsonl').read_text().splitlines()]
                self.assertEqual([r['call_index'] for r in records], [0, 1])
                for record in records:
                    self.assertEqual(record['positions'], list(range(record['prompt_length'], len(record['token_ids']))))

    def test_bad_layer_fails_before_run(self):
        with self.assertRaisesRegex(ValueError, 'has 2 layers'):
            with load_model(self.config(extract_activations=True, activation_layers='2')):
                self.fail('invalid layer accepted')

    def test_answer_capture_matches_full_replay_and_missing_answer(self):
        import torch
        from safetensors.torch import load_file
        cfg = self.config(extract_activations=True,
                          activation_positions='answer-tokens-1,answer-tokens')
        with load_model(cfg) as llm, tempfile.TemporaryDirectory() as out:
            llm.begin_run(out)
            tokens = torch.tensor([[3, 6, 4, 7, 8, 9, 6, 1]])
            recorder = llm.activation_recorder
            recorder.capture(tokens, 3, context={'sample': 0}, seed=1, temperature=0.7)
            recorder.capture(torch.tensor([[3, 6, 4, 8, 1]]), 3,
                             context={'sample': 1}, seed=1, temperature=0.7)
            records = [json.loads(line) for line in (Path(out) / 'activations/index.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['answer_token_index'], 6)
            self.assertEqual(records[0]['positions'], [5, 6])
            cache = load_file(str(Path(out) / 'activations' / records[0]['file']))
            with torch.inference_mode():
                _, full = recorder.bridge.run_with_cache(tokens, names_filter=recorder.names)
            for name, value in cache.items():
                torch.testing.assert_close(value, full[name][:, [5, 6], :], rtol=1e-4, atol=1e-5)
            self.assertIsNone(records[1]['answer_token_index'])
            self.assertIsNone(records[1]['file'])
            self.assertEqual(records[1]['positions'], [])
            self.assertEqual(records[1]['skip_reason'], 'answer_token_not_found')
            del cache, full, value  # Release safetensors mappings before Windows temp cleanup.

    def test_batched_sampling_repeatability_and_remainder(self):
        import torch
        with load_model(self.config()) as llm:
            kwargs = dict(n_samples=5, batch_size=2, seed=19, max_tokens=3, temperature=0.7)
            state = torch.random.get_rng_state().clone()
            with patch.object(llm.model, 'generate', wraps=llm.model.generate) as calls:
                first = list(generate_samples(llm, 'Yes', 'No', **kwargs))
                self.assertEqual([c.kwargs['input_ids'].shape[0] for c in calls.call_args_list], [2, 2, 1])
            self.assertEqual(first, list(generate_samples(llm, 'Yes', 'No', **kwargs)))
            self.assertEqual([i for i, _ in first], list(range(5)))
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            single = generate(llm, 'Yes', 'No', max_tokens=3)
            greedy = list(generate_samples(llm, 'Yes', 'No', n_samples=3, batch_size=3,
                                          max_tokens=3, temperature=0))
            self.assertEqual([text for _, text in greedy], [single] * 3)

    def test_batch_eos_trimming_and_capture_alignment(self):
        import torch
        with load_model(self.config(extract_activations=True, activation_positions='response')) as llm:
            def fixed_generate(input_ids, **kwargs):
                return torch.cat([input_ids, torch.tensor([[6, 1, 2], [7, 6, 1]])], dim=1)
            with tempfile.TemporaryDirectory() as out:
                llm.begin_run(out)
                with patch.object(llm.model, 'generate', side_effect=fixed_generate):
                    list(generate_samples(llm, 'Yes', 'No', n_samples=2, batch_size=2,
                        seed=10, max_tokens=3, activation_context={'id': 'q1', 'condition': 'treatment'}))
                records = [json.loads(line) for line in (Path(out) / 'activations/index.jsonl').read_text().splitlines()]
                self.assertEqual([r['context']['sample'] for r in records], [0, 1])
                self.assertEqual([r['seed'] for r in records], [10, 10])
                self.assertEqual([len(r['positions']) for r in records], [2, 3])
                self.assertEqual([r['token_ids'][-1] for r in records], [1, 1])
                self.assertEqual(llm.performance['generated_tokens'], 5)
                self.assertEqual(llm.performance['token_limit_hits'], 0)

    def test_thinking_mode_passed_to_template(self):
        with load_model(self.config(thinking='off')) as llm:
            with patch.object(llm.tokenizer, 'apply_chat_template', wraps=llm.tokenizer.apply_chat_template) as template:
                generate(llm, 'Yes', 'No', max_tokens=2)
                self.assertIs(template.call_args.kwargs['enable_thinking'], False)

    def test_all_entry_points_record_context(self):
        import torch
        from calibration.run_calibration import main as calibration
        from calibration.calibration_multisample import main as calibration_variance
        from integrity.run_integrity import main as integrity
        from integrity.run_integrity_variance import main as integrity_variance
        questions = self.path / 'questions.json'
        questions.write_text(json.dumps({'questions': [{'id': 'q1', 'question': 'Yes', 'answer': True}]}))

        @contextmanager
        def controlled_model(config):
            with load_model(config) as llm:
                def fixed_generate(input_ids, **kwargs):
                    return torch.cat([input_ids, torch.tensor([[8, 9, 6]]).repeat(input_ids.shape[0], 1)], dim=1)
                # Generate a known parseable answer; extraction still uses real forward passes.
                with patch.object(llm.model, 'generate', side_effect=fixed_generate):
                    yield llm

        for i, main in enumerate((calibration, calibration_variance, integrity, integrity_variance)):
            with self.subTest(entry=i), tempfile.TemporaryDirectory() as out, redirect_stderr(io.StringIO()):
                args = ['--backend', 'transformers', '--model', str(self.path), '--device', 'cpu',
                        '--questions', str(questions), '--out', out, '--max-tokens', '3',
                        '--n-ctx', '512', '--extract-activations']
                if i in (1, 3):
                    args += ['--n-samples', '2']
                if i == 3:
                    args += ['--sample-batch-size', '2']
                with patch('llm.load_model', controlled_model):
                    main(args)
                run = Path(out) / 'run-00'
                self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'complete')
                records = [json.loads(line) for line in (run / 'activations/index.jsonl').read_text().splitlines()]
                self.assertEqual(len(records), (1, 2, 2, 3)[i])
                self.assertTrue(all(r['context']['id'] == 'q1' for r in records))
                if i == 3:
                    self.assertEqual([r['context']['condition'] for r in records], ['control', 'treatment', 'treatment'])
                    self.assertEqual([r['context']['sample'] for r in records], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
