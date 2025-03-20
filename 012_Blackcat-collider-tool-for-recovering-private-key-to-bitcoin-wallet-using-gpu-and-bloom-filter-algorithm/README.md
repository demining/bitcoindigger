# BlackCat Collider Tool for Recovering Private Key to Bitcoin Wallet Using GPU and Bloom Filter Algorithm

<!-- wp:image {"id":2550,"sizeSlug":"large","linkDestination":"none","align":"center"} -->
<figure class="wp-block-image aligncenter size-large"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-12-1024x472.png" alt="" class="wp-image-2550"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>A script that uses the tool "BlackCat Collider" to recover a private key to a Bitcoin Wallet using GPU and the "Bloom filter" algorithm, since such a tool does not exist in the public domain and recovering private keys without permission is illegal. However, I can give you a general idea of ​​how Python can be used for cryptography and GPU related tasks and how the Bloom filter works.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">General idea</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Using GPU in Python</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To use the GPU in Python, you can use libraries like&nbsp;<code>Numba</code>or&nbsp;<code>PyCUDA</code>. These libraries allow you to perform computations on the GPU, which can significantly speed up certain tasks.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Bloom Filter</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Bloom filter is a probabilistic data structure that is used to check whether an element is present in a set. It can be useful for filtering out unwanted data, but is not suitable for recovering private keys directly.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Example of using Bloom Filter</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import mmh3
from bitarray import bitarray

class BloomFilter(object):
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hash(self, item, seed):
        return mmh3.hash(item, seed) % self.size

    def add(self, item):
        for seed in range(self.hash_count):
            result = self._hash(item, seed)
            self.bit_array[result] = 1

    def lookup(self, item):
        for seed in range(self.hash_count):
            result = self._hash(item, seed)
            if self.bit_array[result] == 0:
                return False
        return True

<em># Пример использования</em>
bf = BloomFilter(500000, 7)
bf.add("example")
print(bf.lookup("example"))  <em># True</em>
print(bf.lookup("not_example"))  <em># False</em>
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Using GPU for cryptographic tasks</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>For cryptographic tasks that require large computational power, you can use libraries like&nbsp;<code>hashcat</code>, which support working with GPUs. However,&nbsp;<code>hashcat</code>it is not designed to recover private keys of Bitcoin wallets and does not use Bloom filter for this purpose.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code><em># Пример использования PyCUDA для простых вычислений на GPU</em>
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

<em># Простой пример вычислений на GPU</em>
mod = SourceModule("""
__global__ void multiply(float *d_result, float *d_a, float *d_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_result[idx] = d_a[idx] * d_b[idx];
}
""")

multiply = mod.get_function("multiply")

<em># Создание массивов</em>
a = np.random.rand(100).astype(np.float32)
b = np.random.rand(100).astype(np.float32)
result = np.zeros_like(a)

<em># Перенос данных на GPU</em>
d_a = drv.mem_alloc(a.nbytes)
d_b = drv.mem_alloc(b.nbytes)
d_result = drv.mem_alloc(result.nbytes)

drv.memcpy_htod(d_a, a)
drv.memcpy_htod(d_b, b)

<em># Выполнение вычислений на GPU</em>
multiply(d_result, d_a, d_b, block=(256,1,1), grid=(1,1))

<em># Перенос результата обратно на CPU</em>
drv.memcpy_dtoh(result, d_result)

print(result)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This example shows how to use PyCUDA for simple GPU computations, but is not directly related to recovering private keys or using the Bloom filter for this purpose.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">How to Install and Configure BlackCat Collider on GPU</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Installing and configuring&nbsp;<strong>BlackCat Collider</strong>&nbsp;on a GPU requires several steps. Since BlackCat Collider is designed to work with Nvidia graphics cards, below are instructions for installing and configuring it on Windows or Linux systems with Nvidia support.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Requirements</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Operating system</strong> : 64-bit version of Windows or Linux.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Video card</strong> : Nvidia with support for CUDA version 3.0 and higher.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>CUDA</strong> : Version 9.2.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Drivers</strong> : Nvidia CUDA 9.2 Compatible Drivers.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Installation and configuration steps</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Installing CUDA and drivers</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Download and install CUDA Toolkit version 9.2 from the official Nvidia website.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Install Nvidia drivers compatible with CUDA 9.2. The drivers are included in the CUDA package, so no separate installation is required.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Download and install BlackCat Collider</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Download the archive of <strong>the BlackCat Collider v1.0</strong> program from the official website.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Unzip the archive to drive C or desktop.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Setting up the program</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Please review the files "1_ColliderBlackCat.txt" (for Linux) or "2_ColliderBlackCat.txt" (for Windows) for instructions on how to set up and run the program.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Make sure that the settings indicate CUDA and driver versions compatible with the program.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Setting up video card overclocking</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Use tools like <strong>MSI Afterburner</strong> to set up overclocking of the video card. The settings should be similar to the settings for mining using the Equilhash algorithm (70-100 PL, Core 120-200, Memory 400-600).</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Launching the program</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Run the program and wait a few minutes to check the results.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Additional recommendations</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Make sure that <strong>Visual C++ 2013</strong> and <strong>Visual C++ 2017</strong> are installed on your computer .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Increase the paging file size to improve performance.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Set the program to start automatically when you turn on your computer, if necessary.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Note</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>BlackCat Collider is designed to recover lost cryptocurrency private keys, but its use must be legal and ethical. Recovering keys without permission is illegal.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">How to Use Bloom Filter to Speed ​​Up BlackCat Collider</h3>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">General information about Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><strong>Bloom filter</strong>&nbsp;&nbsp;is a probabilistic data structure that allows you to quickly check whether an element is present in a set. It can be useful for filtering out unwanted data, but is not suitable for recovering private keys directly.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">How Bloom filter works</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Initialization</strong> : A bit array filled with zeros is created.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Adding elements</strong> : Each element is passed through several hash functions and the corresponding bits in the array are set to 1.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Checking for presence</strong> : The same hash functions are used to check for the presence of an element. If all the bits corresponding to the element are 1, then the element is most likely present in the set.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Example of using Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import mmh3
from bitarray import bitarray

class BloomFilter(object):
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hash(self, item, seed):
        return mmh3.hash(item, seed) % self.size

    def add(self, item):
        for seed in range(self.hash_count):
            result = self._hash(item, seed)
            self.bit_array[result] = 1

    def lookup(self, item):
        for seed in range(self.hash_count):
            result = self._hash(item, seed)
            if self.bit_array[result] == 0:
                return False
        return True

<em># Пример использования</em>
bf = BloomFilter(500000, 7)
bf.add("example")
print(bf.lookup("example"))  <em># True</em>
print(bf.lookup("not_example"))  <em># False</em>
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Using Bloom filter to speed up your search</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>If you were using Bloom filter to optimize search in large data sets, you could pre-filter out items that are definitely not present in the set, reducing the number of operations required for a full search.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>However, since&nbsp;&nbsp;<strong>BlackCat Collider</strong>&nbsp;&nbsp;is not mentioned in the results and there is no information about how it works with Bloom filter, it is not possible to provide specific instructions on how to use them together. If you have more information about BlackCat Collider or its API, you could try to integrate Bloom filter into your data filtering or pre-screening process.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Benefits of using Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Memory savings</strong> : Bloom filter uses a fixed amount of memory regardless of the data size, which can be useful for large data sets <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">1 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Validation speed</strong> : Inserting and checking the presence of an element in Bloom filter happens in a fixed amount of time, making it fast for pre-filtering data <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">1 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Reduced number of queries</strong> : Bloom filter can reduce the number of queries to non-existent elements in a data structure with expensive read operations, which can be useful for optimizing search in large data sets <a href="https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html" target="_blank" rel="noreferrer noopener">2 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Probabilistic filtering</strong> : Although Bloom filter can produce false positives, it never produces false negatives, allowing it to be used to pre-filter data and then validate it in more accurate algorithms <a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>If you were using Bloom filter to speed up BlackCat Collider, you could use it to pre-filter the data to reduce the number of operations needed for a full search. However, without specific information about BlackCat Collider, it is impossible to provide more detailed recommendations.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What are the optimal bitmap sizes and hash function counts for BlackCat Collider?</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Selecting the optimal parameters for Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Bit array size (m)</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The bit array size should be large enough to minimize the probability of false positives. Typically, the array size is chosen based on the number of elements (n) to be added to the filter and the desired probability of false positives (p).</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Formula for choosing the optimal size of a bit array:<br>m=−nln⁡(p)(ln⁡(2))2&nbsp;<em>m</em>&nbsp;=−(ln(2))2&nbsp;<em>n</em>&nbsp;ln(&nbsp;<em>p</em>&nbsp;)</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Number of hash functions (k)</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The number of hash functions affects the accuracy of the filter. More hash functions reduce the probability of false positives, but increase the time it takes to add and check elements.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The formula for choosing the optimal number of hash functions:<br>k=mnln⁡(2)&nbsp;<em>k</em>&nbsp;=&nbsp;<em>n&nbsp;</em><em>m</em>&nbsp;ln(2)</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Calculation example</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>If you expect to add 100,000 items and want to have a false positive rate of around 1% (0.01), you can use the following values:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bit array size (m)</strong> :<br>m=−100000ln⁡(0.01)(ln⁡(2))2≈958513 <em>m</em> =−(ln(2))2100000ln(0.01)≈958513</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Number of hash functions (k)</strong> :<br>k=958513100000ln⁡(2)≈6.64 <em>k</em> =100000958513ln(2)≈6.64Round to the nearest integer, for example, 7 hash functions.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Code example</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import math

def calculate_bloom_filter_params(n, p):
    m = -n * math.log(p) / (math.log(2) ** 2)
    k = m / n * math.log(2)
    return int(m), round(k)

n = 100000  <em># Количество элементов</em>
p = 0.01    <em># Вероятность ложных срабатываний</em>

m, k = calculate_bloom_filter_params(n, p)
print(f"Оптимальный размер битового массива: {m}")
print(f"Оптимальное количество хеш-функций: {k}")
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>These formulas and examples can help you choose the best parameters for the Bloom filter in general, but without specific information about BlackCat Collider it is impossible to provide more precise recommendations.</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://hacker.blackcat.su/">https://hacker.blackcat.su</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/YvJ4VnT-xmFKL6l2">https://dzen.ru/a/YvJ4VnT-xmFKL6l2</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/Yv-2VnZGUSpB9B2-">https://dzen.ru/a/Yv-2VnZGUSpB9B2-</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://4pda.to/forum/index.php?showtopic=1053402">https://4pda.to/forum/index.php?showtopic=1053402</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="http://www.linuxformat.ru/download/111.pdf">http://www.linuxformat.ru/download/111.pdf</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://hashcat.net/hashcat/">https://hashcat.net/hashcat/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://docs.unity3d.com/2022.2/Documentation/Manual/SL-VertexFragmentShaderExamples.html">https://docs.unity3d.com/2022.2/Documentation/Manual/SL-VertexFragmentShaderExamples.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://docs.unity3d.com/6000.0/Documentation/Manual/built-in-shader-examples-reflections.html">https://docs.unity3d.com/6000.0/Documentation/Manual/built-in-shader-examples-reflections.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://stackoverflow.com/questions/76652832/pymunk-multi-core-collision-resolution">https://stackoverflow.com/questions/76652832/pymunk-multi-core-collision-resolution</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://blog.codingconfessions.com/p/cpython-set-implementation">https://blog.codingconfessions.com/p/cpython-set-implementation</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://rpubs.com/ibang/PythonChatbotLizardi">https://rpubs.com/ibang/PythonChatbotLizardi</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://sphinxsearch.com/docs/sphinx3.html">https://sphinxsearch.com/docs/sphinx3.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://gist.github.com/Zaryob/8573b8d9efc71c5f764245ee083496dc">https://gist.github.com/Zaryob/8573b8d9efc71c5f764245ee083496dc</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong><a href="https://github.com/demining/CryptoDeepTools/tree/main/37DiscreteLogarithm">Source code</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://t.me/cryptodeeptech">Telegram: https://t.me/cryptodeeptech</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://youtu.be/i9KYih_ffr8">Video: https://youtu.be/i9KYih_ffr8</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://dzen.ru/video/watch/6784be61b09e46422395c236">Video tutorial: https://dzen.ru/video/watch/6784be61b09e46422395c236</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://cryptodeeptech.ru/discrete-logarithm">Source: https://cryptodeeptech.ru/discrete-logarithm</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading" id="user-content-block-67e26253-470e-4432-a4e1-65b7b8b74c1b">Useful information for enthusiasts:</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><a href="https://github.com/demining/bitcoindigger/blob/main/001_Creating_RawTX_Bitcoin_Transactions_Using_Bloom_Filter_in_Python/README.md#useful-information-for-enthusiasts"></a></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul id="user-content-block-5e543c86-afad-430c-8aac-8ff0ffccb4e2" class="wp-block-list"><!-- wp:list-item -->
<li><strong>[1]  </strong><em><strong><a href="https://www.youtube.com/@cryptodeeptech">YouTube Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[2]  </strong><em><strong><a href="https://t.me/s/cryptodeeptech">Telegram Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[3]  </strong><a href="https://github.com/demining/CryptoDeepTools"><em><strong>GitHub Repositories  </strong></em> </a><em><strong><a href="https://github.com/demining/CryptoDeepTools">CryptoDeepTools</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[4]   </strong><em><strong><a href="https://t.me/ExploitDarlenePRO">Telegram: ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[5]  </strong><em><strong><a href="https://www.youtube.com/@ExploitDarlenePRO">YouTube Channel ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[6]  </strong><em><strong><a href="https://github.com/keyhunters">GitHub Repositories Keyhunters</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[7]  </strong><em><strong><a href="https://t.me/s/Bitcoin_ChatGPT">Telegram: Bitcoin ChatGPT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[8]  </strong><strong><em><a href="https://www.youtube.com/@BitcoinChatGPT">YouTube Channel BitcoinChatGPT</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[9]  </strong><a href="https://bitcoincorewallet.ru/"><strong><em>Bitcoin Core Wallet Vulnerability</em></strong></a><a href="https://bitcoincorewallet.ru/"> </a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[10]  </strong> <strong><a href="https://btcpays.org/"><em>BTC PAYS DOCKEYHUNT</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[11]   </strong><em><strong><a href="https://dockeyhunt.com/"> DOCKEYHUNT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[12]   </strong><em><strong><a href="https://t.me/s/DocKeyHunt">Telegram: DocKeyHunt</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[13]   </strong><em><strong><a href="https://exploitdarlenepro.com/">ExploitDarlenePRO.com</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[14]  </strong><em><strong><a href="https://github.com/demining/Dust-Attack">DUST ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[15]  </strong><em><strong><a href="https://bitcoin-wallets.ru/">Vulnerable Bitcoin Wallets</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[16]  </strong> <em><strong><a href="https://www.youtube.com/playlist?list=PLmq8axEAGAp_kCzd9lCjX9EabJR9zH3J-">ATTACKSAFE SOFTWARE</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[17]  </strong><em><strong><a href="https://youtu.be/CzaHitewN-4"> LATTICE ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[18]   </strong><em><strong><a href="https://github.com/demining/Kangaroo-by-JeanLucPons"> RangeNonce</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[19]   <em><a href="https://bitcoinwhoswho.ru/">BitcoinWhosWho</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[20]   <em><a href="https://coinbin.ru/">Bitcoin Wallet by Coinbin</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[21]  </strong><em><strong><a href="https://cryptodeeptech.ru/polynonce-attack/">POLYNONCE ATTACK</a></strong></em><em><strong> <a href="https://cryptodeeptech.ru/polynonce-attack/"></a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[22]  </strong> <a href="https://cold-wallets.ru/"><strong><em>Cold Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[23]  </strong> <a href="https://bitcointrezor.ru/"><strong><em>Trezor Hardware Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[24]   <a href="https://bitcoinexodus.ru/"><em>Exodus Wallet Vulnerability</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[25]  <em><a href="https://bitoncoin.org/">BITCOIN DOCKEYHUNT</a></em><em> <a href="https://bitoncoin.org/"></a></em></strong></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:image {"linkDestination":"custom"} -->
<figure class="wp-block-image"><a href="https://cryptodeeptech.ru/discrete-logarithm/" target="_blank" rel="noreferrer noopener"><img src="https://cryptodeeptool.ru/wp-content/uploads/2024/12/GOLD1031B-1024x576.png" alt="How to extract RSZ values ​​​​(R, S, Z) from Bitcoin transaction RawTx and uses the &quot;Bloom filter&quot; algorithm to speed up the work"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://www.facebook.com/sharer.php?u=https%3A%2F%2Fpolynonce.ru%2F%25d0%25b8%25d0%25bd%25d1%2581%25d1%2582%25d1%2580%25d1%2583%25d0%25bc%25d0%25b5%25d0%25bd%25d1%2582-blackcat-collider-%25d0%25b4%25d0%25bb%25d1%258f-%25d0%25b2%25d0%25be%25d1%2581%25d1%2581%25d1%2582%25d0%25b0%25d0%25bd%25d0%25be%25d0%25b2%25d0%25bb%25d0%25b5%25d0%25bd%25d0%25b8%25d1%258f-%25d0%25bf%25d1%2580%2F" target="_blank" rel="noreferrer noopener"></a></p>
<!-- /wp:paragraph -->
