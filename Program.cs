using System.Drawing;
using System.IO;
using System.Linq;
using System.Globalization;
using back_propagation;

DirectoryInfo di = new DirectoryInfo(System.Environment.CurrentDirectory + @"/mnist_png/training");
List<ImageData> imageDatas = new List<ImageData>();
foreach (var subdir in di.GetDirectories())
{
    Console.WriteLine(subdir.FullName);
    foreach (var file in subdir.GetFiles())
    {
        imageDatas.Add(new ImageData(file.FullName, int.Parse(subdir.Name), int.Parse(file.Name.Split('.')[0])));
    }
}

imageDatas.Sort(((a, b) => { if (a.FileName == b.FileName) return 0; else return (a.FileName < b.FileName) ? 1 : -1; }));

decimal[] tmp = new decimal[128];
decimal[] tmp2 = new decimal[10];
Node[] input = new Node[28 * 28];
Node[] hidden1 = new Node[128];
Node[] hidden2 = new Node[128];
Node[] output = new Node[10];
Random rand = new Random(DateTime.Now.Millisecond);
decimal updateRate = 0.5m;
for (int i = 0; i < 28 * 28; i++)
{ 
    for(int j = 0; j < 128; j++)
    {
        tmp[j] = Convert.ToDecimal(rand.NextDouble()); 
    }
    input[i] = new Node(tmp);
}

for (int i = 0; i < 128; i++)
{
    for (int j = 0; j < 128; j++)
    {
        tmp[j] = Convert.ToDecimal(rand.NextDouble());
    }
    hidden1[i] = new Node(tmp);
}

for (int i = 0; i < 128; i++)
{
    for (int j = 0; j < 10; j++)
    {
        tmp2[j] = Convert.ToDecimal(rand.NextDouble());
    }
    hidden2[i] = new Node(tmp2);
}

for (int i = 0; i < 10; i++)
{
    output[i] = new Node();
}

foreach (var imageData in imageDatas)
{
    Bitmap bitmap = new Bitmap(imageData.ImagePath);

    for (int i = 0; i < bitmap.Width; i++)
    {
        for (int j = 0; j < bitmap.Height; j++)
        {
            Color pixelColor = bitmap.GetPixel(i, j);
            input[i*28+j].x = Convert.ToDecimal(((0.299 * pixelColor.R) + (0.587 * pixelColor.G) + (0.114 * pixelColor.B))/255);
        }
    }
    foreach (var node in hidden1)
    {
        node.Calculate(input, Array.IndexOf(hidden1, node));
    }
    foreach (var node in hidden2)
    {
        node.CalculateReLU(hidden1, Array.IndexOf(hidden2, node));
    }
    foreach (var node in output)
    {
        node.CalculateReLU(hidden2, Array.IndexOf(output, node));
    }
    decimal[] T = new decimal[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    T[imageData.Label] = 1;
    decimal J = 0;
    for(int i = 0; i < 10; i++)
    {
        J += 2 * (output[i].x - T[i]);
    }
    J /= 10.0m;
    foreach (var node in hidden2)
    {
        node.x -= updateRate * J * (node.x > 0 ? node.x : 0);
    }
    foreach (var node in hidden1)
    {
        decimal sum = 0;
        for (int j = 0; j < hidden2.Length; j++)
        {
            sum += J * (hidden2[j].x > 0 ? hidden2[j].x : 0) * node.weights[j];
        }
        node.x -= updateRate * sum * (node.x > 0 ? node.x : 0);
    }
    foreach (var node in input)
    {
        decimal sum = 0;
        for (int h1 = 0; h1 < hidden1.Length; h1++)
        {
            decimal delta_h1 = 0;
            for (int h2 = 0; h2 < hidden2.Length; h2++)
            {
                delta_h1 += J * (hidden2[h2].x > 0 ? hidden2[h2].x : 0) * hidden1[h1].weights[h2];
            }
            sum += delta_h1 * node.weights[h1];
        }
        node.x -= updateRate * sum * (node.x > 0 ? node.x : 0);
    }
    Console.WriteLine($"{imageDatas.IndexOf(imageData)+1}/{imageDatas.Count}");
}
{
    string modelPath = Path.Combine(Environment.CurrentDirectory, "model_weights.txt");
    using (var sw = new StreamWriter(modelPath, false))
    {
        sw.WriteLine("# INPUT");
        foreach (var n in input)
        {
            sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
        }

        sw.WriteLine("# HIDDEN1");
        foreach (var n in hidden1)
        {
            sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
        }

        sw.WriteLine("# HIDDEN2");
        foreach (var n in hidden2)
        {
            sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
        }

        sw.WriteLine("# OUTPUT");
        foreach (var n in output)
        {
            sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
        }
    }
}