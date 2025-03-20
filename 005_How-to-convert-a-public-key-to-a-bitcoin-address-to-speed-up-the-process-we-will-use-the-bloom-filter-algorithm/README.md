# How to convert a public key to a Bitcoin address to speed up the process, we will use the Bloom filter algorithm

<!-- wp:image {"id":2524,"sizeSlug":"large","linkDestination":"none"} -->
<figure class="wp-block-image size-large"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-4-1-1024x567.png" alt="" class="wp-image-2524"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4 class="wp-block-heading"><a href="https://polynonce.ru/author/polynonce/"></a></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Converting a Bitcoin public key to an address can be done using the&nbsp;<code>ecdsa</code>and library&nbsp;<code>base58</code>. However, to speed things up, using the Bloom filter algorithm is not practical in this context, as it is used to filter data in the blockchain, not to convert keys. Below is an example script that converts a public key to legacy and SegWit addresses.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Converting a public key to an address</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import ecdsa
import hashlib
import base58

def hash160(public_key):
    """Хэширование открытого ключа с помощью SHA-256 и RIPEMD-160."""
    return hashlib.new('ripemd160', hashlib.sha256(public_key).digest()).digest()

def public_key_to_address(public_key, compressed=False, segwit=False):
    """Преобразование открытого ключа в адрес."""
    if compressed:
        public_key = public_key[1:]  <em># Удаление префикса 04 для сжатого ключа</em>
    
    <em># Хэширование открытого ключа</em>
    h160 = hash160(public_key)
    
    if segwit:
        <em># Для SegWit адресов используется префикс 'bc1'</em>
        return 'bc1' + base58.b58encode_check(b'\x00' + h160).decode()
    else:
        <em># Для устаревших адресов используется префикс '1'</em>
        return '1' + base58.b58encode_check(b'\x00' + h160).decode()

<em># Пример использования</em>
public_key_uncompressed = b'\x04c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b53f883a9483fb7f2b43f3eacd857c904d1b70ecc168571b64d8f1ab82b57eea88'
public_key_compressed = b'\x02c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b5'

print("Устаревший адрес (несжатый):", public_key_to_address(public_key_uncompressed))
print("Устаревший адрес (сжатый):", public_key_to_address(public_key_compressed, compressed=True))
print("SegWit адрес (несжатый):", public_key_to_address(public_key_uncompressed, segwit=True))
print("SegWit адрес (сжатый):", public_key_to_address(public_key_compressed, compressed=True, segwit=True))
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This script converts a public key to legacy and SegWit addresses, both compressed and uncompressed. The prefix is ​​used for SegWit addresses&nbsp;<code>bc1</code>, and the prefix is ​​used for legacy addresses&nbsp;<code>1</code>.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Note</strong>&nbsp;: To work with real keys, you must ensure that the public key was derived from a valid private key.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What are the alternatives to the Bloom filter algorithm for converting public keys?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><strong>The Bloom Filter</strong>&nbsp;algorithm&nbsp;is not used to convert public keys to addresses. It is a probabilistic data structure designed to quickly check whether an element exists in a data set, but not to transform data. Cryptographic algorithms such as&nbsp;<strong>ECDSA</strong>&nbsp;(Elliptic Curve Digital Signature Algorithm) and hashing with&nbsp;<strong>SHA-256</strong>&nbsp;and&nbsp;<strong>RIPEMD-160</strong>&nbsp;are used to convert Bitcoin public keys to addresses.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>If you're looking for alternatives to optimize your work with datasets that might be useful in other contexts, there are a few options:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Cuckoo Filter</strong> : This is an advanced data structure compared to Bloom Filter that allows elements to be removed and has a lower probability of false positives <a href="https://habr.com/ru/articles/788772/" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Inverse Bloom Filter</strong> : Used to store information about the presence of elements in a set, but with reverse logic.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Hash tables</strong> : Offer a deterministic way to store and search for items, but can be less memory and speed efficient than Bloom filters for very large data sets <a href="https://habr.com/ru/articles/491132/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>There are no alternative algorithms for converting Bitcoin public keys into addresses other than using the cryptographic functions described above.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Example of using a hash table to speed up searches</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Although hash tables are not used directly to transform keys, they can be useful for quickly looking up already transformed addresses:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import hashlib
import base58

<em># Пример хэш-таблицы для быстрого поиска адресов</em>
class HashTable:
    def __init__(self):
        self.table = {}

    def add(self, public_key, address):
        self.table[public_key] = address

    def get(self, public_key):
        return self.table.get(public_key)

<em># Пример использования</em>
hash_table = HashTable()

<em># Преобразование открытого ключа в адрес (аналогично предыдущему примеру)</em>
def public_key_to_address(public_key):
    <em># Реализация преобразования открытого ключа в адрес</em>
    h160 = hashlib.new('ripemd160', hashlib.sha256(public_key).digest()).digest()
    return '1' + base58.b58encode_check(b'\x00' + h160).decode()

public_key = b'\x04c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b53f883a9483fb7f2b43f3eacd857c904d1b70ecc168571b64d8f1ab82b57eea88'
address = public_key_to_address(public_key)

<em># Добавление в хэш-таблицу</em>
hash_table.add(public_key, address)

<em># Быстрый поиск адреса по открытому ключу</em>
print(hash_table.get(public_key))
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This example shows how a hash table can be used to quickly look up addresses that have already been converted from public keys.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What parameters affect the efficiency of Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The effectiveness of a Bloom filter depends on several key parameters:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Filter Size (m)</strong> : This is the number of bits allocated for the filter. The larger the filter size, the less likely it is to generate false positives, but more memory is required to store it <a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">6 </a><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809" target="_blank" rel="noreferrer noopener">7</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Number of hash functions (k)</strong> : This is the number of hash functions that are used to set the bits in the filter. Increasing the number of hash functions reduces the probability of false positives, but increases the processing time <a href="https://habr.com/ru/companies/timeweb/articles/806383/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://habr.com/ru/articles/788772/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Number of elements (n)</strong> : This is the number of unique elements that will be added to the filter. Knowing this number, we can optimize the filter size and the number of hash functions to achieve the desired false positive probability of <a href="https://learn.microsoft.com/ru-ru/azure/databricks/sql/language-manual/delta-create-bloomfilter-index" target="_blank" rel="noreferrer noopener">1 </a><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809" target="_blank" rel="noreferrer noopener">7</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>False Positive Probability (p)</strong> : This is a measure of the accuracy of the filter. The lower the false positive probability, the more accurate the filter, but it may require more memory and computational resources <a href="https://learn.microsoft.com/ru-ru/azure/databricks/sql/language-manual/delta-create-bloomfilter-index" target="_blank" rel="noreferrer noopener">1 </a><a href="https://clickhouse.com/docs/ru/optimize/skipping-indexes" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Quality of hash functions</strong> : Hash functions should provide a uniform distribution of bits to minimize false positives <a href="https://habr.com/ru/articles/788772/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Bloom Filter Optimization Formulas</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To optimize the Bloom filter, you can use the following formulas:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Optimal number of bits (m)</strong> : m=−nln⁡(p)(ln⁡(2))2m = -\frac{n \ln(p)}{(\ln(2))^2}m=−(ln(2))2nln(p)</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Optimal number of hash functions (k)</strong> : k=mnln⁡(2)k = \frac{m}{n} \ln(2)k=nmln(2)</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These formulas allow us to calculate the optimal parameters of the Bloom filter based on the desired false positive rate and the number of elements&nbsp;<a target="_blank" rel="noreferrer noopener" href="https://habr.com/ru/companies/timeweb/articles/806383/">3&nbsp;</a><a target="_blank" rel="noreferrer noopener" href="https://habr.com/ru/articles/788772/">4</a>&nbsp;.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What types of hash functions are best to use in a Bloom filter to minimize false positives</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To minimize false positives in a Bloom filter, it is important to use hash functions that provide a uniform distribution of bits in the bit mask. While there is no single "best" type of hash function for all cases, there are some guidelines:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Cryptographic hash functions</strong> : Although not always necessary for a Bloom filter, they provide a good distribution and can be used if security is a priority. Examples include <strong>SHA-256</strong> or <strong>MD5 </strong><a href="https://kesh.kz/blog/%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80-%D0%91%D0%BB%D1%83%D0%BC%D0%B0-%D0%BD%D0%B0-java/" target="_blank" rel="noreferrer noopener">7</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Non-cryptographic hash functions</strong> : These are faster and are often used in Bloom filters. Examples include <strong>MurmurHash</strong> or <strong>FNV Hash</strong> . These functions are optimized for speed and uniform distribution.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Hash functions with good variance</strong> : These are functions that distribute output values ​​evenly across their range. <strong>MurmurHash</strong> and <strong>CityHash</strong> are popular choices.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Basic requirements for hash functions for Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Uniform distribution</strong> : Hash functions should generate output values ​​that are uniformly distributed across the entire range to minimize collisions.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Independence</strong> : Hash functions should be independent of each other to reduce the chance of simultaneous collisions.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Speed</strong> : Hash functions must be fast so as not to slow down the Bloom filter.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Example of using MurmurHash in Python</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import mmh3

def murmur_hash(data, seed):
    return mmh3.hash(data, seed)

<em># Пример использования</em>
data = "example_data"
seeds = [1, 2, 3]  <em># Используйте разные семена для разных хэш-функций</em>

hash_values = [murmur_hash(data, seed) for seed in seeds]
print(hash_values)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This example shows how to use&nbsp;<strong>MurmurHash</strong>&nbsp;with different seeds to generate multiple hash values ​​for the same input, which is useful for Bloom filters.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://2012.nscf.ru/Tesis/Vasilev.pdf">https://2012.nscf.ru/Tesis/Vasilev.pdf</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
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
<li><a href="https://otus.ru/nest/post/972/">https://otus.ru/nest/post/972/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://kesh.kz/blog/%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80-%D0%91%D0%BB%D1%83%D0%BC%D0%B0-%D0%BD%D0%B0-java/">https://kesh.kz/blog/%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80-%D0%91%D0%BB%D1%83%D0%BC%D0%B0-%D0%BD%D0%B0-java/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://learn.microsoft.com/ru-ru/azure/databricks/sql/language-manual/delta-create-bloomfilter-index">https://learn.microsoft.com/ru-ru/azure/databricks/sql/language-manual/delta-create-bloomfilter-index</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://2012.nscf.ru/Tesis/Vasilev.pdf">https://2012.nscf.ru/Tesis/Vasilev.pdf</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/timeweb/articles/806383/">https://habr.com/ru/companies/timeweb/articles/806383/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://clickhouse.com/docs/ru/optimize/skipping-indexes">https://clickhouse.com/docs/ru/optimize/skipping-indexes</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading"></h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/491132/">https://habr.com/ru/articles/491132/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.k0d.cc/storage/books/CRYPTO/%D0%91%D1%80%D1%8E%D1%81%20%D0%A8%D0%BD%D0%B0%D0%B9%D0%B5%D1%80%20-%20%D0%9F%D1%80%D0%B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D0%B0%D1%8F%20%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F/7.PDF">https://www.k0d.cc/storage/books/CRYPTO/%D0%91%D1%80%D1%8E%D1%81%20%D0%A8%D0%BD%D0%B0%D0%B9%D0%B5%D1%80%20-%20%D0%9F%D1%80%D0 %B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D0%B0%D1%8F%20%D0%BA%D1%80%D0 %B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F/7.PDF</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://joschi.gitlab.io/telegram-clickhouse-archive/clickhouse_ru/2021-04_2.html">https://joschi.gitlab.io/telegram-clickhouse-archive/clickhouse_ru/2021-04_2.html</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading"></h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/">https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/YsR8D0FFcSzXhY9v">https://dzen.ru/a/YsR8D0FFcSzXhY9v</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.stackoverflow.com/questions/64496/%D0%A8%D0%B8%D1%84%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D1%81-%D0%BE%D1%82%D0%BA%D1%80%D1%8B%D1%82%D1%8B%D0%BC-%D0%BA%D0%BB%D1%8E%D1%87%D0%BE%D0%BC">https://ru.stackoverflow.com/questions/64496/%D0%A8%D0%B8%D1%84%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8 %D0%B5-%D1%81-%D0%BE%D1%82%D0%BA%D1%80%D1%8B%D1%82%D1%8B%D0%BC-%D0%BA%D0%BB%D1%8E%D1%87%D0%BE%D0%BC</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/564256/">https://habr.com/ru/articles/564256/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitnovosti.io/2020/07/05/blokchejn-glossarij-terminov/">https://bitnovosti.io/2020/07/05/blokchejn-glossarij-terminov/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://miningclub.info/threads/keyhunter-py-poisk-privatkey-bitcion-do-2012-na-otformatirovannyx-diskax.31532/">https://miningclub.info/threads/keyhunter-py-poisk-privatkey-bitcion-do-2012-na-otformatirovannyx-diskax.31532/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82">https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82</a></li>
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
