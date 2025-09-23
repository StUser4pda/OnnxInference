using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;

namespace OnnxInference
{
    class Program
    {
        // Define paths to your ONNX models and vocab files
        private static readonly string VocabPath = "vocabs.json";
        private static readonly string EncoderModelPath = "encoder.onnx";
        private static readonly string DecoderModelPath = "decoder.onnx";

        private static Dictionary<char, long> letter2id;
        private static Dictionary<string, long> ph2id;
        private static Dictionary<long, string> id2ph;

        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: OnnxInference.exe <input_file> <output_file>");
                return;
            }

            string inputFile = args[0];
            string outputFile = args[1];

            if (!File.Exists(inputFile))
            {
                Console.WriteLine("Invalid input file.");
                return;
            }
            Console.WriteLine("Loading vocabularies...");
            LoadVocabularies(VocabPath);
            Console.WriteLine("Vocabularies loaded.");

            Console.WriteLine("Initializing ONNX sessions...");
            using (var encSession = new InferenceSession(EncoderModelPath))
            using (var decSession = new InferenceSession(DecoderModelPath))
            {
                Console.WriteLine("ONNX sessions initialized.");
                Console.WriteLine($"Processing words from {inputFile}...");
                ProcessFile(inputFile, outputFile, encSession, decSession);
                Console.WriteLine($"Finished. Output saved to {outputFile}");
            }
        }

        private static void ProcessFile(string inputFile, string outputFile, InferenceSession encSession, InferenceSession decSession)
        {
            try
            {
                var lines = File.ReadLines(inputFile, Encoding.UTF8);
                using (var writer = new StreamWriter(outputFile, false, Encoding.UTF8))
                {
                    foreach (var line in lines)
                    {
                        string word = line.Trim();
                        if (string.IsNullOrEmpty(word))
                            continue;

                        string normalizedWord = Regex.Replace(word, @"\+(.)", "$1+");
                        string phonemes = InferWord(normalizedWord, encSession, decSession, maxLen: 50, beamSize: 5);
                        writer.WriteLine($"{word}\t{phonemes}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }
        }

        private class Beam
        {
            public double LogProb;
            public List<int> Tokens;
            public Tensor<float> Hidden;
            public Tensor<float> Cell;
        }

        private static string InferWord(string word, InferenceSession encSession, InferenceSession decSession, int maxLen = 50, int beamSize = 5)
        {
            // Encode input word
            var wordIds = word.Select(c => letter2id.ContainsKey(c) ? letter2id[c] : 0).ToList();
            var srcInput = new DenseTensor<long>(wordIds.ToArray(), new[] { 1, wordIds.Count });
            var srcLenInput = new DenseTensor<long>(new long[] { wordIds.Count }, new[] { 1 });

            var encInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("src_input", srcInput),
                NamedOnnxValue.CreateFromTensor("src_len_input", srcLenInput)
            };

            var encOutputs = encSession.Run(encInputs);
            var encoderHidden = encOutputs.First(v => v.Name == "encoder_hidden").AsTensor<float>();
            var encoderCell = encOutputs.First(v => v.Name == "encoder_cell").AsTensor<float>();

            int sosId = (int)(ph2id.ContainsKey("<sos>") ? ph2id["<sos>"] : 1);
            int eosId = (int)(ph2id.ContainsKey("<eos>") ? ph2id["<eos>"] : ph2id.Values.Max());

            // Initialize beams with <sos>
            var beams = new List<Beam>
            {
                new Beam { LogProb = 0.0, Tokens = new List<int> { sosId }, Hidden = encoderHidden, Cell = encoderCell }
            };
            var completed = new List<Beam>();

            for (int step = 0; step < maxLen; step++)
            {
                var allCandidates = new List<Beam>();

                foreach (var beam in beams)
                {
                    int lastToken = beam.Tokens.Last();
                    if (lastToken == eosId)
                    {
                        completed.Add(beam);
                        continue;
                    }

                    var decInputTensor = new DenseTensor<long>(new long[] { lastToken }, new[] { 1 });
                    var decInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input_token", decInputTensor),
                        NamedOnnxValue.CreateFromTensor("decoder_hidden_in", beam.Hidden),
                        NamedOnnxValue.CreateFromTensor("decoder_cell_in", beam.Cell)
                    };

                    var decOutputs = decSession.Run(decInputs);
                    var outputLogits = decOutputs.First(v => v.Name == "output_logits").AsTensor<float>().ToArray();

                    // Convert logits -> log probabilities
                    double maxLogit = outputLogits.Max();
                    var exp = outputLogits.Select(x => Math.Exp(x - maxLogit)).ToArray();
                    double sumExp = exp.Sum();
                    var logProbs = exp.Select(x => Math.Log(x / sumExp)).ToArray();

                    // Take top-k
                    var topk = logProbs
                        .Select((lp, idx) => new { Idx = idx, LogP = lp })
                        .OrderByDescending(x => x.LogP)
                        .Take(beamSize);

                    foreach (var candidate in topk)
                    {
                        var newHidden = decOutputs.First(v => v.Name == "decoder_hidden_out").AsTensor<float>();
                        var newCell = decOutputs.First(v => v.Name == "decoder_cell_out").AsTensor<float>();

                        allCandidates.Add(new Beam
                        {
                            LogProb = beam.LogProb + candidate.LogP,
                            Tokens = beam.Tokens.Concat(new[] { candidate.Idx }).ToList(),
                            Hidden = newHidden,
                            Cell = newCell
                        });
                    }
                }

                if (!allCandidates.Any()) break;

                beams = allCandidates.OrderByDescending(b => b.LogProb).Take(beamSize).ToList();

                if (beams.All(b => b.Tokens.Last() == eosId))
                {
                    completed.AddRange(beams);
                    break;
                }
            }

            var best = completed.Any()
                ? completed.OrderByDescending(b => b.LogProb).First()
                : beams.First();

            // Convert token ids -> phoneme string
            var sb = new StringBuilder();
            foreach (var token in best.Tokens.Skip(1)) // skip sos
            {
                if (token == eosId) break;
                if (id2ph.TryGetValue(token, out var ph))
                    sb.Append(ph);
            }
            return sb.ToString();
        }

        private static void LoadVocabularies(string path)
        {
            try
            {
                var jsonText = File.ReadAllText(path, Encoding.UTF8);
                var vocabData = JsonConvert.DeserializeObject<VocabData>(jsonText);

                letter2id = vocabData.letter2id
                    .Where(kvp => kvp.Key.Length == 1)
                    .ToDictionary(kvp => kvp.Key[0], kvp => kvp.Value);

                ph2id = vocabData.ph2id;
                id2ph = vocabData.id2ph.ToDictionary(kvp => long.Parse(kvp.Key), kvp => kvp.Value);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load vocabularies: {ex.Message}");
                throw;
            }
        }
    }

    public class VocabData
    {
        public Dictionary<string, long> letter2id { get; set; }
        public Dictionary<string, long> ph2id { get; set; }
        public Dictionary<string, string> id2ph { get; set; }
    }
}
