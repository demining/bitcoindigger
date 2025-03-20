# How to get X and Y coordinates from a Bitcoin public key to speed up the process we will use the Bloom filter algorithm

<!-- wp:image {"id":2527,"sizeSlug":"large","linkDestination":"none"} -->
<figure class="wp-block-image size-large"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-5-1024x520.png" alt="" class="wp-image-2527"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Below is a Python script that gets the X and Y coordinates of a Bitcoin public key. This script uses an&nbsp;<code>ecdsa</code>elliptic curve library and does not include an implementation of the Bloom filter algorithm, as it is not directly applicable to getting coordinates from a public key. Bloom filters are typically used to quickly filter out items in large data sets, but not for cryptographic computations.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import ecdsa
from ecdsa.curves import SECP256k1

def get_coordinates(public_key):
    <em># Установка кривой SECP256k1</em>
    curve = SECP256k1
    
    <em># Загрузка открытого ключа</em>
    if public_key.startswith('04'):  <em># Несжатый формат</em>
        public_key_bytes = bytes.fromhex(public_key[2:])
    elif public_key.startswith('02') or public_key.startswith('03'):  <em># Сжатый формат</em>
        public_key_bytes = bytes.fromhex(public_key[2:])
        <em># Для сжатого формата нам нужно восстановить полный ключ</em>
        <em># Это требует дополнительных шагов, которые не поддерживаются напрямую в ecdsa</em>
        <em># Для простоты будем использовать только несжатый формат</em>
        print("Сжатые ключи не поддерживаются в этом примере.")
        return None
    else:
        print("Неправильный формат открытого ключа.")
        return None

    <em># Создание объекта открытого ключа</em>
    vk = ecdsa.VerifyingKey.from_string(public_key_bytes, curve=curve)

    <em># Получение координат X и Y</em>
    x = hex(vk.public_key.point.x)[2:]  <em># Удаление '0x'</em>
    y = hex(vk.public_key.point.y)[2:]  <em># Удаление '0x'</em>

    <em># Десятичные значения</em>
    x_dec = vk.public_key.point.x
    y_dec = vk.public_key.point.y

    return x, y, x_dec, y_dec

<em># Пример использования</em>
public_key = "04c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b53f883a9483fb7f2b43f3eacd857c904d1b70ecc168571b64d8f1ab82b57eea88"
coords = get_coordinates(public_key)

if coords:
    x_hex, y_hex, x_dec, y_dec = coords
    print(f"Координаты X (шестнадцатеричный): {x_hex}")
    print(f"Координаты Y (шестнадцатеричный): {y_hex}")
    print(f"Координаты X (десятичный): {x_dec}")
    print(f"Координаты Y (десятичный): {y_dec}")
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This script only works with uncompressed public key formats. Compressed keys require additional processing to recover the full key, which is not directly supported by the library&nbsp;<code>ecdsa</code>.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Note:</strong>&nbsp;The Bloom filter implementation is not applicable in this context, as it is used for quickly filtering elements in large data sets, not for cryptographic computations.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">How to use Bloom filter to speed up work with public key coordinates</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Using&nbsp;<strong>a Bloom filter</strong>&nbsp;to speed up Bitcoin public key coordinates is not a straightforward solution, as the Bloom filter is designed to quickly check whether an element belongs to a set, not to perform cryptographic computations. However, if you want to quickly check whether certain public keys are present in your dataset, a Bloom filter can be useful.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Example of using Bloom filter to check for public keys</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Creating a Bloom filter:</strong><!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Determine the filter size ( <code>size</code>) and the number of hash functions ( <code>hash_count</code>).</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Initialize the bit array.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Adding public keys to the filter:</strong><!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Apply hash functions to each public key and set the corresponding bits in the filter.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Checking for the presence of a public key:</strong><!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Apply the same hash functions to the public key being verified.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>If all the corresponding bits are set, the key is probably present in the set.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Python code example</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import hashlib
import mmh3

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size

    def _hash(self, item, seed):
        return mmh3.hash(item, seed) % self.size

    def add(self, item):
        for seed in range(self.hash_count):
            index = self._hash(item, seed)
            self.bit_array[index] = True

    def check(self, item):
        for seed in range(self.hash_count):
            index = self._hash(item, seed)
            if not self.bit_array[index]:
                return False
        return True

<em># Пример использования</em>
bloom_filter = BloomFilter(1000, 5)

<em># Добавление открытых ключей в фильтр</em>
public_keys = ["04c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b53f883a9483fb7f2b43f3eacd857c904d1b70ecc168571b64d8f1ab82b57eea88"]
for key in public_keys:
    bloom_filter.add(key)

<em># Проверка наличия открытого ключа</em>
print(bloom_filter.check(public_keys[0]))  <em># True</em>
print(bloom_filter.check("03anotherkey"))  <em># False</em>
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This example shows how to use Bloom filter to quickly check for public keys in a dataset. However, you still need to use cryptographic libraries such as&nbsp;<code>ecdsa</code>. Bloom filter does not speed up the process of calculating the coordinates, but it can help in managing datasets.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading"><br>What are the alternatives to the Bloom filter algorithm to speed up work with coordinates?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>If you are looking for alternatives to the&nbsp;<strong>Bloom filter</strong>&nbsp;algorithm to speed up work with Bitcoin public key coordinates, here are a few approaches and data structures that may be useful depending on the specific task:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Cuckoo Filter</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Cuckoo filter is an improved version of Bloom filter that allows you to remove items from the filter, which is not supported by the traditional Bloom filter. This can be useful if you need to not only check for items, but also remove them.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Usage:</strong> Suitable for scenarios where you need to quickly check and remove elements from a data set.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Bitmap Index</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> A Bitmap index is a data structure that allows you to quickly filter data by multiple binary attributes. This is especially effective for queries that involve multiple conditions.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Usage:</strong> Used in databases to quickly filter data by multiple columns with low cardinality.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Hash Tables</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Hash tables are data structures that allow you to quickly add, remove, and check the existence of elements in a data set.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Suitable for scenarios where you need to quickly check for the presence of elements and perform operations on them.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Trie (Prefix tree)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> A Trie is a tree where each node is a prefix of a string. This allows you to quickly check for the presence of strings in a data set.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Usage:</strong> Used to quickly search and check for rows in a data set.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Roaring Bitmaps</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Roaring bitmaps are optimized bitmaps that use a combination of bitmaps, arrays, and bit sequences to efficiently store and search data.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Usage:</strong> Used on large datasets for fast filtering and searching by multiple attributes.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Each of these data structures has its own advantages and disadvantages, and the choice depends on the specific requirements of your task.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Sample Code for Cuckoo Filter in Python</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import mmh3

class CuckooFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.buckets = [[] for _ in range(size)]

    def _hash(self, item, seed):
        return mmh3.hash(item, seed) % self.size

    def add(self, item):
        for seed in range(self.hash_count):
            index = self._hash(item, seed)
            if len(self.buckets[index]) &lt; 2:
                self.buckets[index].append(item)
                return True
        <em># Если все места заняты, необходимо удалить один элемент</em>
        <em># Реализация удаления не показана для простоты</em>

    def check(self, item):
        for seed in range(self.hash_count):
            index = self._hash(item, seed)
            if item in self.buckets[index]:
                return True
        return False

<em># Пример использования</em>
cuckoo_filter = CuckooFilter(1000, 5)
cuckoo_filter.add("04c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b53f883a9483fb7f2b43f3eacd857c904d1b70ecc168571b64d8f1ab82b57eea88")
print(cuckoo_filter.check("04c5389a31ce6149c28ba20d14db8540b2319e5a65000a2919fbf7a6296e7840b53f883a9483fb7f2b43f3eacd857c904d1b70ecc168571b64d8f1ab82b57eea88"))  <em># True</em>
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This example shows how you can use the Cuckoo filter to quickly check if elements exist in a dataset.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">When Bitmap Indexes Can Be More Efficient Than Bloom Filters</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><strong>Bitmap indexes</strong>&nbsp;can be more efficient than&nbsp;<strong>Bloom filters</strong>&nbsp;in the following cases:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Queries with logical operations</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Bitmap indexes are well suited for queries that involve logical operations ( <code>AND</code>, <code>OR</code>, <code>XOR</code>) on multiple columns. They allow such operations to be performed quickly by combining bitmaps of corresponding values <a href="https://habr.com/ru/companies/badoo/articles/451938/" target="_blank" rel="noreferrer noopener">​​1 </a><a href="https://citforum.ru/database/oracle/bb_indexes/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Suitable for decision support systems (DSS) where complex queries with multiple conditions are often executed.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>High cardinality</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Although bitmap indexes have traditionally been considered suitable only for low-cardinality fields, modern implementations have shown them to be effective for high-cardinality fields as well <a href="https://habr.com/ru/companies/badoo/articles/451938/" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Used in cases where it is necessary to quickly filter data by several columns, regardless of the number of unique values.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Search accuracy</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Unlike Bloom filters, which allow false positives, bitmap indexes provide accurate searching without additional checks <a href="https://habr.com/ru/companies/postgrespro/articles/349224/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://postgrespro.ru/docs/postgresql/16/bloom" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Suitable for applications where search accuracy is of paramount importance.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Combining queries</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Bitmap indexes allow you to efficiently combine queries across multiple columns, making them more flexible for complex queries <a href="https://citforum.ru/database/oracle/bb_indexes/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%B0%D1%86%D0%B8%D1%8F_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85._%D0%94%D1%80%D1%83%D0%B3%D0%B8%D0%B5_%D1%82%D0%B8%D0%BF%D1%8B_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BE%D0%B2._%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BE%D0%B2" target="_blank" rel="noreferrer noopener">7</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Used in databases to optimize complex queries.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Decision Support Systems (DSS)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> In DSS systems where queries are frequently executed during off-peak periods and do not require high data refresh rates, bitmap indexes can be more efficient due to their ability to quickly process complex queries <a href="https://citforum.ru/database/oracle/bb_indexes/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Suitable for analytical systems where query speed is critical.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>However, it should be noted that bitmap indexes may be less efficient in online transaction processing (OLTP) systems, where frequent data updates can lead to locking issues and slow index updates&nbsp;<a target="_blank" rel="noreferrer noopener" href="https://citforum.ru/database/oracle/bb_indexes/">2</a>&nbsp;.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading"><br>What are the most efficient Bitcoin public key storage formats?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Bitcoin public key storage formats can vary depending on specific security and usability requirements. Below are some of the most efficient public key storage formats and methods:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Uncompressed format</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> The uncompressed format of the public key starts with a prefix <code>04</code>and contains 65 bytes (130 characters in hexadecimal notation). It includes both X and Y coordinates.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Usage:</strong> Used for full public key identification, but less efficient in size.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Compressed format</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> The compressed public key format starts with the prefix <code>02</code>or <code>03</code>and contains 33 bytes (66 characters in hexadecimal notation). It includes only the X coordinate and the sign of the Y coordinate.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> More compact and widely used in modern applications.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Storage in wallets</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> Public keys are often stored in cryptocurrency wallets, which can be software (e.g. Electrum), hardware (e.g. Ledger), or paper wallets.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Suitable for convenient management and use of public keys in transactions.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Storage in SegWit format (Bech32)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> SegWit addresses use the Bech32 protocol and start with the prefix <code>bc1</code>. They are more efficient and support compressed public keys.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Used to reduce transaction sizes and fees.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Advantages of each format:</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Uncompressed format:</strong> Full key identification.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Compressed format:</strong> Compact and efficient.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Wallets:</strong> Ease of use and security.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>SegWit (Bech32):</strong> Efficiency and Fee Reduction.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>The choice of format depends on specific security requirements, usability, and data size.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What are the cryptographically strong hash functions for Bitcoin?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The Bitcoin blockchain uses several cryptographically strong hash functions to ensure data security and integrity. The main hash functions used in Bitcoin include:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>SHA-256 (Secure Hash Algorithm 256-bit)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> SHA-256 is a widely used cryptographic hash function that converts input data of arbitrary length into a 256-bit hash code. It provides high security and is used to generate block and transaction hashes in the Bitcoin blockchain <a href="https://belapan.by/cryptocurrency/kakaya-heshiruyushhaya-funkcziya-ispolzuetsya-v-blokchejne-bitkojna" target="_blank" rel="noreferrer noopener">1 </a><a href="https://woolypooly.com/ru/blog/khesh-kriptografiya-moshchnyj-dvigatel" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Application:</strong> Used to ensure the integrity and security of data in the blockchain, as well as for mining.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.RIPEMD&nbsp;<strong>-160</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description:</strong> RIPEMD-160 is a hash function that converts input data into a 160-bit hash code. In Bitcoin, it is used in combination with SHA-256 to create addresses <a href="https://habr.com/ru/articles/319868/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Usage:</strong> Used to generate Bitcoin addresses, providing a compact and secure representation of public keys.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Applications of hash functions in Bitcoin:</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>SHA-256:</strong> Used for block and transaction hashing, as well as for mining.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>RIPEMD-160:</strong> Used to generate Bitcoin addresses from public keys.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These hash functions provide cryptographic strength and data integrity to the Bitcoin blockchain, which is a key aspect of its security.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://belapan.by/cryptocurrency/kakaya-heshiruyushhaya-funkcziya-ispolzuetsya-v-blokchejne-bitkojna">https://belapan.by/cryptocurrency/kakaya-heshiruyushhaya-funkcziya-ispolzuetsya-v-blokchejne-bitkojna</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://woolypooly.com/ru/blog/khesh-kriptografiya-moshchnyj-dvigatel">https://woolypooly.com/ru/blog/khesh-kriptografiya-moshchnyj-dvigatel</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.wikipedia.org/wiki/%D0%9A%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%85%D0%B5%D1%88-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F">https://ru.wikipedia.org/wiki/%D0%9A%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8% D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%85%D0%B5%D1%88-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/319868/">https://habr.com/ru/articles/319868/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ubiklab.net/posts/cryptographic-hash-functions/">https://ubiklab.net/posts/cryptographic-hash-functions/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.khanacademy.org/video/bitcoin-cryptographic-hash-function">https://ru.khanacademy.org/video/bitcoin-cryptographic-hash-function</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://sky.pro/wiki/javascript/kriptograficheskie-hesh-funkcii-sha-1-sha-2-sha-3-i-bezopasnost/">https://sky.pro/wiki/javascript/kriptograficheskie-hesh-funkcii-sha-1-sha-2-sha-3-i-bezopasnost/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://intuit.ru/studies/courses/3643/885/lecture/32299">https://intuit.ru/studies/courses/3643/885/lecture/32299</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://www.binance.com/ru/blog/ecosystem/%D0%BF%D1%83%D0%B1%D0%BB%D0%B8%D1%87%D0%BD%D1%8B%D0%B5-%D0%BA%D0%BB%D1%8E%D1%87%D0%B8-%D0%B8-%D0%BF%D1%80%D0%B8%D0%B2%D0%B0%D1%82%D0%BD%D1%8B%D0%B5-%D0%BA%D0%BB%D1%8E%D1%87%D0%B8-%D1%87%D1%82%D0%BE-%D1%8D%D1%82%D0%BE-%D0%B8-%D0%BA%D0%B0%D0%BA-%D0%BE%D0%BD%D0%B8-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%D1%8E%D1%82-421499824684903332">https://www.binance.com/ru/blog/ecosystem/%D0%BF%D1%83%D0%B1%D0%BB%D0%B8%D1%87%D0%BD%D1%8B%D 0%B5-%D0%BA%D0%BB%D1%8E%D1%87%D0%B8-%D0%B8-%D0%BF%D1%80%D0%B8%D0%B2%D0%B0%D1%82%D0%BD%D1%8B%D 0%B5-%D0%BA%D0%BB%D1%8E%D1%87%D0%B8-%D1%87%D1%82%D0%BE-%D1%8D%D1%82%D0%BE-%D0%B8-%D0%BA%D0%B0 %D0%BA-%D0%BE%D0%BD%D0%B8-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%D1%8E%D1%82-421499824684903332</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://coinsutra.com/ru/store-private-keys/">https://coinsutra.com/ru/store-private-keys/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/530676/">https://habr.com/ru/articles/530676/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://sunscrypt.ru/blog/goryachie-koshelki/vybor-koshelka-dlya-kriptovalyuty/">https://sunscrypt.ru/blog/goryachie-koshelki/vybor-koshelka-dlya-kriptovalyuty/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forklog.com/cryptorium/kakie-formaty-byvayut-u-bitkoin-adresov">https://forklog.com/cryptorium/kakie-formaty-byvayut-u-bitkoin-adresov</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ataix.kz/blog/article/where-to-store-bitcoins">https://ataix.kz/blog/article/where-to-store-bitcoins</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://coinsutra.com/ru/bitcoin-private-key/">https://coinsutra.com/ru/bitcoin-private-key/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.ledger.com/ru/academy/segwit-%D0%B8-native-segwit-bech32-%D0%BA%D0%B0%D0%BA%D0%B0%D1%8F-%D0%BC%D0%B5%D0%B6%D0%B4%D1%83-%D0%BD%D0%B8%D0%BC%D0%B8-%D1%80%D0%B0%D0%B7%D0%BD%D0%B8%D1%86%D0%B0">https://www.ledger.com/ru/academy/segwit-%D0%B8-native-segwit-bech32-%D0%BA%D0%B0%D0%BA%D0%B0%D1%8F -%D0%BC%D0%B5%D0%B6%D0%B4%D1%83-%D0%BD%D0%B8%D0%BC%D0%B8-%D1%80%D0%B0%D0%B7%D0%BD%D0%B8%D1%86%D0%B0</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/badoo/articles/451938/">https://habr.com/ru/companies/badoo/articles/451938/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://citforum.ru/database/oracle/bb_indexes/">https://citforum.ru/database/oracle/bb_indexes/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/postgrespro/articles/349224/">https://habr.com/ru/companies/postgrespro/articles/349224/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://docs.tantorlabs.ru/tdb/ru/16_6/be/bloom.html">https://docs.tantorlabs.ru/tdb/ru/16_6/be/bloom.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://postgrespro.ru/docs/postgresql/16/bloom">https://postgrespro.ru/docs/postgresql/16/bloom</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://postgrespro.ru/docs/postgresql/9.6/bloom">https://postgrespro.ru/docs/postgresql/9.6/bloom</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%B0%D1%86%D0%B8%D1%8F_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85._%D0%94%D1%80%D1%83%D0%B3%D0%B8%D0%B5_%D1%82%D0%B8%D0%BF%D1%8B_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BE%D0%B2._%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BE%D0%B2">https://neerc.ifmo.ru/wiki/index.php?title=%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%B0%D1%8 6%D0%B8%D1%8F_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85._%D0%94%D1%80%D1%83%D0%B3%D0%B8%D0%B5_%D 1%82%D0%B8%D0%BF%D1%8B_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BE%D0%B2._%D0%9F%D1%80%D0%B8 %D0%BC%D0%B5%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BE%D0%B2</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://docs.tantorlabs.ru/tdb/ru/16_8/be/bloom.html">https://docs.tantorlabs.ru/tdb/ru/16_8/be/bloom.html</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/badoo/articles/451938/">https://habr.com/ru/companies/badoo/articles/451938/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://fit-old.nsu.ru/data_/docs/mag/OOP/4_RPD/KM/_KM_DV1.4_rpd.pdf">https://fit-old.nsu.ru/data_/docs/mag/OOP/4_RPD/KM/_KM_DV1.4_rpd.pdf</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://evmservice.ru/blog/filtr-bluma/">https://evmservice.ru/blog/filtr-bluma/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/XZR3S9W7wwCsGo3R">https://dzen.ru/a/XZR3S9W7wwCsGo3R</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/304800/">https://habr.com/ru/articles/304800/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://docs.tantorlabs.ru/pipelinedb/1.2/probabilistic.html">https://docs.tantorlabs.ru/pipelinedb/1.2/probabilistic.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.k0d.cc/storage/books/CRYPTO/%D0%91%D1%80%D1%8E%D1%81%20%D0%A8%D0%BD%D0%B0%D0%B9%D0%B5%D1%80%20-%20%D0%9F%D1%80%D0%B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D0%B0%D1%8F%20%D0%BA%D1%80%D0%B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F/7.PDF">https://www.k0d.cc/storage/books/CRYPTO/%D0%91%D1%80%D1%8E%D1%81%20%D0%A8%D0%BD%D0%B0%D0%B9%D0%B5%D1%80%20-%20%D0%9F%D1%80%D0 %B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D0%B0%D1%8F%20%D0%BA%D1%80%D0 %B8%D0%BF%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8F/7.PDF</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://postgrespro.ru/media/docs/postgresql/13/ru/postgres-A4.pdf">https://postgrespro.ru/media/docs/postgresql/13/ru/postgres-A4.pdf</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/319868/">https://habr.com/ru/articles/319868/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://cryptodeep.ru/rowhammer-attack/">https://cryptodeep.ru/rowhammer-attack/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/564256/">https://habr.com/ru/articles/564256/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://coinspot.io/technology/bitkojn-vvedenie-dlya-razrabotchikov-2/">https://coinspot.io/technology/bitkojn-vvedenie-dlya-razrabotchikov-2/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82">https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pikabu.ru/story/uyazvimost_deserializesignature_v_seti_bitkoin_kriptoanaliz_posledstviya_i_vozmozhnost_sozdaniya_nedeystvitelnyikh_podpisey_ecdsa_11454555">https://pikabu.ru/story/uyazvimost_deserializesignature_v_seti_bitkoin_kriptoanaliz_posledstviya_i_vozmozhnost_sozdaniya_nedeystvitelnyikh_podpisey_ecdsa_11454555</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/svtrostov/oclexplorer/issues/6">https://github.com/svtrostov/oclexplorer/issues/6</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://21ideas.org/epubs/programming-bitcoin.pdf">https://21ideas.org/epubs/programming-bitcoin.pdf</a></li>
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

<!-- wp:paragraph -->
<p><a href="https://www.facebook.com/sharer.php?u=https%3A%2F%2Fpolynonce.ru%2F%25d0%25ba%25d0%25b0%25d0%25ba-%25d0%25bf%25d0%25be%25d0%25bb%25d1%2583%25d1%2587%25d0%25b8%25d1%2582%25d1%258c-%25d0%25ba%25d0%25be%25d0%25be%25d1%2580%25d0%25b4%25d0%25b8%25d0%25bd%25d0%25b0%25d1%2582%25d1%258b-x-%25d0%25b8-y-%25d0%25b8%25d0%25b7-%25d0%25bf%25d1%2583%25d0%25b1%25d0%25bb%25d0%25b8%25d1%2587%25d0%25bd%2F" target="_blank" rel="noreferrer noopener"></a></p>
<!-- /wp:paragraph -->
