# Which Nvidia video cards are used to find a private key to a Bitcoin Wallet using the Bloom filter algorithm

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:image {"lightbox":{"enabled":false},"id":2539,"sizeSlug":"large","linkDestination":"custom"} -->
<figure class="wp-block-image size-large"><a href="https://cryptodeeptech.ru/discrete-logarithm" target="_blank" rel=" noreferrer noopener"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-9-1-1024x714.png" alt="" class="wp-image-2539"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>A Python script that uses Nvidia graphics cards to find a Bitcoin wallet private key using the Bloom filter algorithm requires several steps. First, we need to use a library that allows working with GPUs, such as&nbsp;<strong>Numba</strong>&nbsp;or&nbsp;<strong>PyCUDA</strong>&nbsp;. Second, implementing the Bloom filter algorithm to find private keys is not a standard task, since Bloom filters are usually used to quickly determine the presence of an element in a set, rather than to find keys.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>However, we can use the GPU to speed up private key searches, which is a more typical approach. Below is an example script that uses PyCUDA to speed up private key searches. This script does not directly use the Bloom filter, but it demonstrates how the GPU can be used to speed up the calculations.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 1: Installing the required libraries</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Install PyCUDA and other required libraries:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">bash<code>pip install pycuda numpy
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 2: Python Script</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import hashlib

<em># Функция для вычисления хеша RIPEMD-160</em>
def ripemd160(data):
    return hashlib.new('ripemd160', data).digest()

<em># CUDA Кернел для поиска приватных ключей</em>
mod = SourceModule("""
__global__ void search_private_keys(unsigned char *hashes, int num_hashes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx &gt;= num_hashes) return;

    unsigned char private_key[32];
    unsigned char public_key[65];
    unsigned char hash[20];

    // Примерная реализация поиска приватного ключа
    // В реальности это будет более сложный процесс
    for (int i = 0; i &lt; 32; i++) {
        private_key[i] = idx + i;
    }

    // Вычисление публичного ключа из приватного (упрощенная версия)
    // В реальности используется эллиптическая криптография
    for (int i = 0; i &lt; 65; i++) {
        public_key[i] = private_key[i % 32];
    }

    // Вычисление хеша RIPEMD-160 для публичного ключа
    // В реальности используется SHA-256 + RIPEMD-160
    for (int i = 0; i &lt; 20; i++) {
        hash[i] = public_key[i % 65];
    }

    // Сравнение с заданными хешами
    for (int i = 0; i &lt; num_hashes; i++) {
        if (memcmp(hash, &amp;hashes[i*20], 20) == 0) {
            printf("Found private key: ");
            for (int j = 0; j &lt; 32; j++) {
                printf("%02x", private_key[j]);
            }
            printf("\\n");
        }
    }
}
""")

<em># Получение функции из модуля</em>
search_private_keys = mod.get_function("search_private_keys")

<em># Примерные данные для поиска</em>
num_hashes = 10
hashes = np.zeros((num_hashes, 20), dtype=np.uint8)

<em># Копирование данных на GPU</em>
d_hashes = drv.mem_alloc(hashes.nbytes)
drv.memcpy_htod(d_hashes, hashes)

<em># Запуск CUDA ядра</em>
search_private_keys(d_hashes, np.int32(num_hashes), block=(256,1,1), grid=(1,1))

<em># Очистка</em>
drv.Context.synchronize()
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Notes:</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>This script is a simplified example</strong> and is not intended for real private key mining. Implementing private key mining requires a deep understanding of cryptography and elliptic curves.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Bloom Filter</strong> is typically used to quickly determine the presence of an element in a set, rather than to find keys. For finding private keys, brute force or other methods such as Pollard's Kangaroo are more efficient.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Elliptic curve cryptography</strong> requires specialized libraries, such as <code>ecdsa</code>or <code>secp256k1</code>, to compute public keys from private keys.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>This script demonstrates the basic use of PyCUDA to accelerate GPU computations, but for real private key mining a more complex and specialized solution is needed.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">What security measures should be taken when using Bloom filter to find private keys</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>When using&nbsp;<strong>Bloom filter</strong>&nbsp;to find private keys, there are several security measures to consider to minimize risks and ensure data security:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Safety when using Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Probability of false positive results</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Limitations of Bloom Filter</strong> : Bloom filter can produce false positives, meaning it may report the presence of a private key that is not actually present. This can lead to unnecessary actions or false alarms <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Measures to reduce false positives</strong> : Increasing the bit array size and the number of hash functions can reduce the probability of false positives <a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://evmservice.ru/blog/filtr-bluma/" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Data protection</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Data Storage</strong> : Even if a Bloom filter does not store the private keys themselves, it may contain information that can be used to recover the keys. Therefore, it is important to protect the filter itself and all data associated with it <a href="https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Encryption</strong> : Use encryption to protect data associated with the Bloom Filter, especially if it contains sensitive information <a href="https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Inability to delete elements</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Restrictions on deletion</strong> : Bloom filter does not allow deleting elements without changing the data structure. This means that if a private key has been added to a filter, it cannot be deleted without completely refreshing the filter <a href="https://habr.com/ru/articles/788772/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Data management measures</strong> : Use additional mechanisms to track added and removed keys outside the Bloom filter.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Error handling</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Handling False Positives</strong> : Develop procedures to check and validate the results obtained from the Bloom Filter to avoid acting on false positives <a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">4 </a><a href="https://evmservice.ru/blog/filtr-bluma/" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Confidentiality</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Privacy protection</strong> : Ensure that the Bloom filter does not reveal information about specific private keys, even if they are present in filter <a href="https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">6.&nbsp;<strong>Update and support</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Regular Updates</strong> : Update and test your Bloom Filter regularly to ensure it remains effective and secure.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Using Bloom Filter to find private keys requires careful security considerations to minimize the risks associated with false positives and data protection. It is important to understand the limitations of Bloom Filter and implement additional security measures to protect sensitive information.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading"></h3>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What steps do you need to take to configure Nvidia graphics cards for computing acceleration?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To configure your Nvidia graphics card to accelerate computations, such as using CUDA for cryptographic tasks, follow these steps:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Steps to set up Nvidia graphics card</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Updating drivers</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Update your drivers</strong> : Make sure your graphics card drivers are updated to the latest version. This can be done through the Nvidia Control Panel or on the official Nvidia website.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Configuring CUDA</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Manage 3D settings</strong> : Open the Nvidia Control Panel and go to Manage 3D settings <a href="https://blog.eldorado.ru/publications/kak-nastroit-videokartu-nvidia-i-poluchit-maksimalnyy-fps-21925" target="_blank" rel="noreferrer noopener">3 </a><a href="https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/Manage_3D_Settings_(reference).htm" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>CUDA - GPU</strong> : In this section, select the graphics processor (GPU) to use for CUDA applications. Make sure that the GPU selected is the one you want to use to accelerate computations <a href="https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/Manage_3D_Settings_(reference).htm" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Performance optimization</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Maximum Performance Mode</strong> : In Settings, select Maximum Performance mode for the GPU to ensure maximum performance during computing <a href="https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/Manage_3D_Settings_(reference).htm" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Dynamic Boost</strong> : Enable Dynamic Boost to allow the system to dynamically distribute power consumption between the GPU and CPU for optimal performance <a href="https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/Manage_3D_Settings_(reference).htm" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Using monitoring and tuning tools</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>MSI Afterburner</strong> : Use tools like MSI Afterburner to monitor GPU temperatures and loads, and adjust clock speeds and voltages for optimal performance and reduced heat <a href="https://hyperpc.ru/blog/gaming-pc/video-card-undervolt" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Checking the functionality</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Testing</strong> : Test your graphics card in applications that use CUDA to ensure all settings are correct and performance is optimal.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These steps will help you configure your Nvidia graphics card to accelerate computing using CUDA technology. Be sure to keep up to date with driver updates and adjust the settings based on the specific requirements of your applications.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">What are some alternatives to Pollard's Kangaroo method for finding private keys?</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>There are several alternatives to&nbsp;<strong>Pollard's Kangaroo</strong>&nbsp;method for finding private keys in cryptography, especially in the context of elliptic curves:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Alternatives to Pollard's Kangaroo Method</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Pollard's Rho Method</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : This method is also used to calculate discrete logarithms in cyclic groups. It is less efficient than the kangaroo method, but can be useful in certain scenarios.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Ease of implementation and relatively low memory requirements.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Baby-Step Giant-Step (BSGS)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : This algorithm is also designed to calculate discrete logarithms. It works by dividing the range into "small steps" and "large steps", which allows it to efficiently search for a solution.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Faster than Pollard's Rho method, but requires more memory.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Factorization methods</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Although not directly applicable to finding private keys on elliptic curves, factorization methods (such as Shors's or Pollard's rho methods for factorization) can be used to solve related problems.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Can be effective for certain types of cryptographic problems.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Quantum computing</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Uses quantum computers to solve discrete logarithm problems, which can potentially be much faster than classical methods.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Theoretically can provide exponential speedup compared to classical algorithms.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Endomorphism</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Used to speed up elliptic curve operations such as ECDSA signature verification. Although not a direct alternative for finding private keys, it can be used to optimize related computations <a href="https://cryptodeep.ru/endomorphism/" target="_blank" rel="noreferrer noopener">7</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Speeds up calculations on elliptic curves.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Conclusion</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Each of these methods has its own advantages and disadvantages, and the choice depends on the specific task and available resources. Pollard's Kangaroo method remains one of the most effective for finding private keys in a known range due to its efficiency and parallelizability&nbsp;<a target="_blank" rel="noreferrer noopener" href="https://habr.com/ru/articles/679626/">3&nbsp;</a><a target="_blank" rel="noreferrer noopener" href="https://cryptodeep.ru/kangaroo/">4</a>&nbsp;.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">What are the most efficient scalar multiplication methods on elliptic curves?</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The most efficient methods of scalar multiplication on elliptic curves are:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Efficient methods for scalar multiplication</h2>
<!-- /wp:heading -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Binary Non-False Function (NAF) Method</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Uses a binary representation of a scalar, where adjacent bits are always different. This reduces the number of addition operations needed for scalar multiplication <a href="http://novtex.ru/prin/rus/10.17587/prin.7.21-28.html" target="_blank" rel="noreferrer noopener">1 </a><a href="https://istina.msu.ru/download/18836198/1f8xOB:3dpBGuZW21zrcHt6Du5DwrLk5Tg/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Reduces the number of addition operations, making it more efficient than the simple binary algorithm.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>wNAF (wide non-joint representation) method</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Extends the NAF method by allowing wider windows to be used to represent a scalar, further reducing the number of operations <a href="https://istina.msu.ru/download/18836198/1f8xOB:3dpBGuZW21zrcHt6Du5DwrLk5Tg/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : More efficient than NAF, especially for large www values.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Montgomery Ladder</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Performs scalar multiplication using only doubling and addition operations, making it resistant to side-channel attacks <a href="https://ru.ruwiki.ru/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%9C%D0%BE%D0%BD%D1%82%D0%B3%D0%BE%D0%BC%D0%B5%D1%80%D0%B8_(%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5_%D0%BA%D1%80%D0%B8%D0%B2%D1%8B%D0%B5)" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Secure and efficient, especially in cryptographic applications.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>RTNAF (Restricted Torsion NAF) method</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Operating principle</strong> : Used for Koblitz curves and allows optimizing operations on these curves <a href="https://www.mathnet.ru/php/getFT.phtml?jrnid=cheb&amp;paperid=316&amp;what=fullt" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Efficient for curves with characteristic 2, making it useful in certain cryptographic applications.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Mixed coordinates</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>How it works</strong> : Uses a combination of affine and projective coordinates to optimize scalar multiplication operations <a href="https://www.mathnet.ru/php/getFT.phtml?jrnid=cheb&amp;paperid=316&amp;what=fullt" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Benefits</strong> : Can provide significant productivity gains, especially for larger field characteristic values.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Each of these methods has its own advantages and disadvantages, and the choice depends on the specific requirements of the application, such as security, performance, and memory constraints.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">What Koblitz curves are used in the NIST standard</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The NIST standard for elliptic curve cryptography uses curves recommended by Jerry Solinas (NSA) based on the work of Neil Koblitz. However, Koblitz curves, which are a special type of elliptic curve optimized for efficient implementation, are not directly mentioned in the NIST recommendations.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>NIST recommends using curves such as&nbsp;<strong>P-192</strong>&nbsp;,&nbsp;<strong>P-224</strong>&nbsp;,&nbsp;<strong>P-256</strong>&nbsp;,&nbsp;<strong>P-384</strong>&nbsp;, and&nbsp;<strong>P-521</strong>&nbsp;, which are defined over prime fields and are not Koblitz curves. These curves are chosen for their security and implementation efficiency&nbsp;<a target="_blank" rel="noreferrer noopener" href="https://ru.wikipedia.org/wiki/%D0%AD%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F">1&nbsp;</a><a target="_blank" rel="noreferrer noopener" href="https://www.ssl.com/ru/%D1%81%D1%82%D0%B0%D1%82%D1%8C%D1%8E/%D1%87%D1%82%D0%BE-%D1%82%D0%B0%D0%BA%D0%BE%D0%B5-%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F-%D1%81-%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9-%D0%BA%D1%80%D0%B8%D0%B2%D0%BE%D0%B9-ecc/">6</a>&nbsp;.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Koblitz curves are commonly used in other contexts, such as optimizing operations on elliptic curves over fields of characteristic 2, which can be useful in certain applications, but they are not on the NIST recommended list for cryptographic protocols.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://ru.wikipedia.org/wiki/%D0%AD%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F">https://ru.wikipedia.org/wiki/%D0%AD%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81 %D0%BA%D0%B0%D1%8F_%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://libeldoc.bsuir.by/bitstream/123456789/52756/1/Patyupin_Vvedenie.pdf">https://libeldoc.bsuir.by/bitstream/123456789/52756/1/Patyupin_Vvedenie.pdf</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/191240/">https://habr.com/ru/articles/191240/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.wikipedia.org/wiki/ECDSA">https://ru.wikipedia.org/wiki/ECDSA</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://publications.hse.ru/pubs/share/direct/316478376.pdf">https://publications.hse.ru/pubs/share/direct/316478376.pdf</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.ssl.com/ru/%D1%81%D1%82%D0%B0%D1%82%D1%8C%D1%8E/%D1%87%D1%82%D0%BE-%D1%82%D0%B0%D0%BA%D0%BE%D0%B5-%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F-%D1%81-%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9-%D0%BA%D1%80%D0%B8%D0%B2%D0%BE%D0%B9-ecc/">https://www.ssl.com/ru/%D1%81%D1%82%D0%B0%D1%82%D1%8C%D1%8E/%D1%87%D1%82%D0%B E-%D1%82%D0%B0%D0%BA%D0%BE%D0%B5-%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1 %80%D0%B0%D1%84%D0%B8%D1%8F-%D1%81-%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8 %D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9-%D0%BA%D1%80%D0%B8%D0%B2%D0%BE%D0%B9-ecc/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/news/836094/">https://habr.com/ru/news/836094/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.osp.ru/os/2002/07-08/181696">https://www.osp.ru/os/2002/07-08/181696</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="http://novtex.ru/prin/rus/10.17587/prin.7.21-28.html">http://novtex.ru/prin/rus/10.17587/prin.7.21-28.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.mathnet.ru/php/getFT.phtml?jrnid=cheb&amp;paperid=316&amp;what=fullt">https://www.mathnet.ru/php/getFT.phtml?jrnid=cheb&amp;paperid=316&amp;what=fullt</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.wikipedia.org/wiki/%D0%90%D1%82%D0%B0%D0%BA%D0%B0_%D0%BF%D0%BE_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B0%D0%BC_%D0%B2%D1%8B%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D0%B9_%D0%BD%D0%B0_%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5_%D0%BA%D1%80%D0%B8%D0%B2%D1%8B%D0%B5,_%D0%B8%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D1%83%D1%8E%D1%89%D0%B8%D0%B5_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%9C%D0%BE%D0%BD%D1%82%D0%B3%D0%BE%D0%BC%D0%B5%D1%80%D0%B8">https://ru.wikipedia.org/wiki/%D0%90%D1%82%D0%B0%D0%BA%D0%B0_%D0%BF%D0%BE_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B0%D0%BC_%D0%B 2%D1%8B%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D0%B9_%D0%B D%D0%B0_%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5 %D1%81%D0%BA%D0%B8%D0%B5_%D0%BA%D1%80%D0%B8%D0%B2%D1%8B%D0%B5 ,_%D0%B8%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D1%83%D1%8E%D1%8 9%D0%B8%D0%B5_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%B C_%D0%9C%D0%BE%D0%BD%D1%82%D0%B3%D0%BE%D0%BC%D0%B5%D1%80%D0%B8</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://moluch.ru/archive/15/1426/">https://moluch.ru/archive/15/1426/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://istina.msu.ru/download/18836198/1f8xOB:3dpBGuZW21zrcHt6Du5DwrLk5Tg/">https://istina.msu.ru/download/18836198/1f8xOB:3dpBGuZW21zrcHt6Du5DwrLk5Tg/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.ruwiki.ru/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%9C%D0%BE%D0%BD%D1%82%D0%B3%D0%BE%D0%BC%D0%B5%D1%80%D0%B8_(%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5_%D0%BA%D1%80%D0%B8%D0%B2%D1%8B%D0%B5)">https://ru.ruwiki.ru/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%9C%D0%BE%D0%BD%D1%82%D0%B3%D0%BE%D0%BC%D0%B5%D1 %80%D0%B8_(%D1%8D%D0%BB%D0%BB%D0%B8%D0%BF%D1%82%D0%B8%D1%87%D0% B5%D1%81%D0%BA%D0%B8%D0%B5_%D0%BA%D1%80%D0%B8%D0%B2%D1%8B%D0%B5)</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://cyberleninka.ru/article/n/effektivnye-algoritmy-skalyarnogo-umnozheniya-v-gruppe-tochek-ellipticheskoy-krivoy">https://cyberleninka.ru/article/n/effektivnye-algoritmy-skalyarnogo-umnozheniya-v-gruppe-tochek-ellipticheskoy-krivoy</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/335906/">https://habr.com/ru/articles/335906/</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://dzen.ru/a/Yva3FLB_PRpSz1wz">https://dzen.ru/a/Yva3FLB_PRpSz1wz</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/682220/">https://habr.com/ru/articles/682220/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/679626/">https://habr.com/ru/articles/679626/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://cryptodeep.ru/kangaroo/">https://cryptodeep.ru/kangaroo/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/svtrostov/oclexplorer/issues/6">https://github.com/svtrostov/oclexplorer/issues/6</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pikabu.ru/@CryptoDeepTech?mv=2&amp;page=4&amp;page=6">https://pikabu.ru/@CryptoDeepTech?mv=2&amp;page=4&amp;page=6</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://cryptodeep.ru/endomorphism/">https://cryptodeep.ru/endomorphism/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://nvsu.ru/ru/materialyikonf/1572/Kultura,%20nauka,%20obrazovanie%20-%20problemi%20i%20perspektivi%20-%20CH.1.%20-%20Mat%20konf%20-%202015.pdf">https://nvsu.ru/ru/materialyikonf/1572/Kultura,%20nauka,%20obrazovanie%20-%20problemi%20i%20perspektivi%20-%20CH.1.%20-%20Mat%20konf%20-%202015.pdf</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/To_adjust_3D_hardware_acceleration.htm">https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/To_adjust_3D_hardware_acceleration.htm</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ya.ru/neurum/c/tehnologii/q/kak_nastroit_videokartu_nvidia_dlya_dostizheniya_afd6942e">https://ya.ru/neurum/c/tehnologii/q/kak_nastroit_videokartu_nvidia_dlya_dostizheniya_afd6942e</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://blog.eldorado.ru/publications/kak-nastroit-videokartu-nvidia-i-poluchit-maksimalnyy-fps-21925">https://blog.eldorado.ru/publications/kak-nastroit-videokartu-nvidia-i-poluchit-maksimalnyy-fps-21925</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://hyperpc.ru/blog/gaming-pc/video-card-undervolt">https://hyperpc.ru/blog/gaming-pc/video-card-undervolt</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.ixbt.com/live/sw/nastroyka-videokarty-nvidia-dlya-raboty-i-igr.html">https://www.ixbt.com/live/sw/nastroyka-videokarty-nvidia-dlya-raboty-i-igr.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/Manage_3D_Settings_(reference).htm">https://www.nvidia.com/content/Control-Panel-Help/vLatest/ru-ru/mergedProjects/nv3dRUS/Manage_3D_Settings_(reference).htm</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://club.dns-shop.ru/blog/t-99-videokartyi/84377-panel-upravleniya-nvidia-polnyii-obzor-vozmojnostei/">https://club.dns-shop.ru/blog/t-99-videokartyi/84377-panel-upravleniya-nvidia-polnyii-obzor-vozmojnostei/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.mvideo.ru/blog/pomogaem-razobratsya/kak-nastroit-panel-upravleniya-nvidia-dlya-igr-obzor-klyuchevyh-parametrov">https://www.mvideo.ru/blog/pomogaem-razobratsya/kak-nastroit-panel-upravleniya-nvidia-dlya-igr-obzor-klyuchevyh-parametrov</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike">https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html">https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://evmservice.ru/blog/filtr-bluma/">https://evmservice.ru/blog/filtr-bluma/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.crust.irk.ru/images/upload/newsabout241/1191.pdf">https://www.crust.irk.ru/images/upload/newsabout241/1191.pdf</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/682220/">https://habr.com/ru/articles/682220/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/684362/">https://habr.com/ru/articles/684362/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://miningclub.info/threads/keyhunter-py-poisk-privatkey-bitcion-do-2012-na-otformatirovannyx-diskax.31532/">https://miningclub.info/threads/keyhunter-py-poisk-privatkey-bitcion-do-2012-na-otformatirovannyx-diskax.31532/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/svtrostov/oclexplorer">https://github.com/svtrostov/oclexplorer</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/OxideDevX/btcbruter_script">https://github.com/OxideDevX/btcbruter_script</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.programmersforum.ru/showthread.php?t=327290">https://www.programmersforum.ru/showthread.php?t=327290</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.youtube.com/watch?v=nCegyIXr_b0">https://www.youtube.com/watch?v=nCegyIXr_b0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcointalk.org/index.php?topic=2827744.0">https://bitcointalk.org/index.php?topic=2827744.0</a></li>
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
<figure class="wp-block-image"><a href="https://cryptodeeptech.ru/discrete-logarithm/" target="_blank" rel="noreferrer noopener"><img src="https://camo.githubusercontent.com/3568257c92c143826ea99b9e75ccdf3b05ed58e05081f33d93b5aca5129d67fc/68747470733a2f2f63727970746f64656570746f6f6c2e72752f77702d636f6e74656e742f75706c6f6164732f323032342f31322f474f4c4431303331422d31303234783537362e706e67" alt="Discrete Logarithm mathematical methods and tools for recovering cryptocurrency wallets Bitcoin"/></a></figure>
<!-- /wp:image -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->
