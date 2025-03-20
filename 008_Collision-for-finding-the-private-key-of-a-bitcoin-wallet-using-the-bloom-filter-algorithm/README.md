# Collision for finding the private key of a Bitcoin wallet using the Bloom filter algorithm

<!-- wp:image {"lightbox":{"enabled":false},"id":2536,"sizeSlug":"large","linkDestination":"custom","align":"center"} -->
<figure class="wp-block-image aligncenter size-large"><a href="https://cryptodeeptech.ru/discrete-logarithm" target="_blank" rel=" noreferrer noopener"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-9-1024x1003.png" alt="" class="wp-image-2536"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>A Python script that uses collisions to find a Bitcoin wallet's private key using the Bloom filter algorithm is impractical and almost impossible for several reasons:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Collision Probability</strong> : The probability of finding a collision for the RIPEMD-160 hash function used in Bitcoin is extremely low. Collisions exist for most hash functions, but their frequency is close to the theoretical minimum of <a href="https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D0%BB%D0%BB%D0%B8%D0%B7%D0%B8%D1%8F_%D1%85%D0%B5%D1%88-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%B8" target="_blank" rel="noreferrer noopener">4 </a><a href="https://habr.com/ru/articles/319868/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Keyspace</strong> : The private keyspace in Bitcoin is huge (2^256), making brute-force or collision searching nearly impossible without massive computing resources <a href="https://habr.com/ru/articles/319868/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Bloom filter</strong> : The Bloom filter algorithm is used to quickly determine the presence of an element in a set, but it is not suitable for finding collisions or private keys directly.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>However, if you want to understand how you can use the GPU to enumerate keys (without using the Bloom filter), you can look at examples such as the "oclexplorer" project, which uses OpenCL to accelerate key enumeration&nbsp;<a target="_blank" rel="noreferrer noopener" href="https://github.com/svtrostov/oclexplorer">1</a>&nbsp;.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Below is an example of a simple private key enumeration using Python and a library&nbsp;<code>ecdsa</code>for computing public keys and hashes. This example does not use a Bloom filter and is not intended for practical use due to the huge key space.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import hashlib
import ecdsa
import random

def generate_private_key():
    """Генерирует случайный приватный ключ."""
    return ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)

def get_public_key(private_key):
    """Получает публичный ключ из приватного."""
    return private_key.get_verifying_key()

def get_ripemd160_hash(public_key):
    """Вычисляет хеш RIPEMD-160 для публичного ключа."""
    sha256_hash = hashlib.sha256(public_key.to_string()).digest()
    return hashlib.new('ripemd160', sha256_hash).hexdigest()

def search_collision(target_hash):
    """Простой пример поиска коллизии."""
    while True:
        private_key = generate_private_key()
        public_key = get_public_key(private_key)
        hash_value = get_ripemd160_hash(public_key)
        if hash_value == target_hash:
            print("Коллизия найдена!")
            return private_key

<em># Пример использования</em>
target_hash = "some_target_hash_here"  <em># Замените на целевой хеш</em>
search_collision(target_hash)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This script is not intended for real collision detection due to its low efficiency and huge key space. For practical purposes, it is recommended to use existing Bitcoin libraries and tools, such as&nbsp;<code>bitcoinlib</code>or&nbsp;<code>bit</code>, to create and manage wallets&nbsp;<a target="_blank" rel="noreferrer noopener" href="https://habr.com/ru/articles/525638/">2&nbsp;</a><a target="_blank" rel="noreferrer noopener" href="https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python">7</a>&nbsp;.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What are the alternatives to Bloom filter algorithm for finding collisions?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Alternatives to the&nbsp;<strong>Bloom filter</strong>&nbsp;algorithm for finding collisions or checking whether an element belongs to a set include the following data structures and algorithms:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Hash tables</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Advantages</strong> : Hash tables provide an accurate check for whether an element belongs to a set, but require more memory, especially when working with large data sets.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Disadvantages</strong> : Hash tables can experience collisions, requiring additional resolution mechanisms such as chained hashing or open <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">6</a> addressing .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Cuckoo hashing</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Advantages</strong> : Cuckoo hashing is designed to resolve collisions in hash tables and provides constant worst-case retrieval time. It is more memory efficient than traditional hash tables, but can be more complex to implement <a href="https://ru.wikipedia.org/wiki/%D0%9A%D1%83%D0%BA%D1%83%D1%88%D0%BA%D0%B8%D0%BD%D0%BE_%D1%85%D0%B5%D1%88%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Disadvantages</strong> : May be less efficient if the table is very full, resulting in increased time to insert and delete elements <a href="https://ru.wikipedia.org/wiki/%D0%9A%D1%83%D0%BA%D1%83%D1%88%D0%BA%D0%B8%D0%BD%D0%BE_%D1%85%D0%B5%D1%88%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Quotient Filter</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Advantages</strong> : Quotient Filter is similar to Bloom filter in speed and complexity, but has a false negative rate, meaning it may report an element as missing even if it is present. It uses more memory than Bloom filter <a href="https://rdl-journal.ru/article/download/746/821/1185" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Disadvantages</strong> : Performance degrades as the filter fills, making it less effective for very large datasets <a href="https://rdl-journal.ru/article/download/746/821/1185" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Representation trees</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Advantages</strong> : Representation trees, such as suffix trees or prefix trees, can be used to find substrings or elements in a set. They provide accurate data checking, but can be more complex to implement <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Disadvantages</strong> : Typically used for string data and may be less efficient for other data types.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Counting Bloom Filter</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Advantages</strong> : This is a variation of the Bloom filter that allows you to track the number of times each element occurs, which can help reduce false positives <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Disadvantages</strong> : Requires more memory than the standard Bloom filter and may be less efficient for very large datasets.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>The choice of alternative depends on the specific requirements of the task, such as the need for accuracy, memory limitations, and data processing speed.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What security measures should be taken into account when using Bloom filter to find private keys</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>When using&nbsp;<strong>Bloom filter</strong>&nbsp;to find private keys or any sensitive data, the following security precautions should be taken into account:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Collision protection</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Increasing the filter size</strong> : The larger the filter size, the lower the probability of false positive results <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Increasing the number of hash functions</strong> : Using multiple hash functions reduces the likelihood of collisions <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Choosing good hash functions</strong> : Use hash functions with low collision probability <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Data encryption</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Encrypted Bloom Filters</strong> : Use encryption to protect the data in the filter, as is done in some systems <a href="https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://bigdataschool.ru/blog/bloom-filter-for-parquet-files-in-spark-apps.html" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Data anonymization</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Anonymous Identifiers</strong> : Use anonymized identifiers to protect personal information, as in the BreachWatch <a href="https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/" target="_blank" rel="noreferrer noopener">2</a> system .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Location of data processing</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Client-side processing</strong> : Process sensitive data on the client side to reduce the risk of leakage <a href="https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Filter update</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Update your filter regularly to keep it current and effective, especially if your data set changes frequently <a href="https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Access control</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Restrict access to the filter and data to prevent unauthorized use or information leakage.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Implementation of additional measures</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Use additional security measures such as secure cryptographic functions and hardware security modules (HSMs) to protect data <a href="https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These measures will help minimize the risks associated with using Bloom filter for sensitive data. However, it is worth noting that finding private keys using Bloom filter is not a practical approach due to the huge key space and the likelihood of false positives.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">What are the ready-made tools for finding Bitcoin private keys</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>There are several ready-made tools and services that can be used to find Bitcoin private keys, although most of them are intended for theoretical or educational purposes rather than practical use due to the huge key space. Here are some examples:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>SecretScan</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>This is a pool for finding private keys of Bitcoin, Ethereum and other cryptocurrencies. SecretScan uses GPU to speed up the search process and allows users to participate in the pool to jointly find keys <a href="https://secretscan.ru/" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>BTC Puzzle</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Provides a complete list of all Bitcoin private keys, but this is more of a theoretical tool as the probability of finding a used address is extremely low <a href="https://btcpuzzle.info/ru/private-keys/btc" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>BlackCat Collider</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>This is a password recovery tool for encrypted Bitcoin wallets. It uses multi-threaded processes to generate and verify keys, but is not designed to find random private keys <a href="https://wsem.ru/publications/bitcoin_koshelek_3335/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://4pda.to/forum/index.php?showtopic=1053402" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Key Hunter - Bitcoin Checker</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>An Android app that calculates Bitcoin and Ethereum wallet addresses from private keys and checks the balance. It can also search for wallets with positive balances in the background <a href="https://play.google.com/store/apps/details?id=io.github.keyhunter" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Metasploit Framework and MSFVenom</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>They are used to find vulnerabilities in code, but can be used to extract private keys from vulnerable applications such as some mobile wallets <a href="https://habr.com/ru/articles/773412/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These tools are mainly used for educational or research purposes, since finding random private keys is almost impossible due to the huge key space (2^256).</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://secretscan.ru/">https://secretscan.ru</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://btcpuzzle.info/ru/private-keys/btc">https://btcpuzzle.info/ru/private-keys/btc</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://wsem.ru/publications/bitcoin_koshelek_3335/">https://wsem.ru/publications/bitcoin_koshelek_3335/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/773412/">https://habr.com/ru/articles/773412/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://play.google.com/store/apps/details?id=io.github.keyhunter">https://play.google.com/store/apps/details?id=io.github.keyhunter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://4pda.to/forum/index.php?showtopic=1053402">https://4pda.to/forum/index.php?showtopic=1053402</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/distributedlab/articles/413627/">https://habr.com/ru/companies/distributedlab/articles/413627/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.bellingcat.com/materialy/putevoditeli/2017/10/31/bitcoin-follow-the-trail/">https://ru.bellingcat.com/materialy/putevoditeli/2017/10/31/bitcoin-follow-the-trail/</a></li>
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
<li><a href="https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/">https://www.keepersecurity.com/blog/ru/2018/10/29/introducing-keeper-breachwatch/</a></li>
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
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://evmservice.ru/blog/filtr-bluma/">https://evmservice.ru/blog/filtr-bluma/</a></li>
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
<li><a href="https://rdl-journal.ru/article/download/746/821/1185">https://rdl-journal.ru/article/download/746/821/1185</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="http://e-notabene.ru/view_article.php?id_article=30541&amp;nb=1&amp;logged=0&amp;aurora=0">http://e-notabene.ru/view_article.php?id_article=30541&amp;nb=1&amp;logged=0&amp;aurora=0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A5%D0%B5%D1%88%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%BA%D1%83%D0%BA%D1%83%D1%88%D0%BA%D0%B8">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A5%D0%B5%D1%88%D0%B8%D1%80%D0 %BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%BA%D1%83%D0%BA%D1%83%D1%88%D0%BA%D0%B8</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.wikipedia.org/wiki/%D0%9A%D1%83%D0%BA%D1%83%D1%88%D0%BA%D0%B8%D0%BD%D0%BE_%D1%85%D0%B5%D1%88%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5">https://ru.wikipedia.org/wiki/%D0%9A%D1%83%D0%BA%D1%83%D1%88%D0%BA%D0%B8%D0 %BD%D0%BE_%D1%85%D0%B5%D1%88%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://github.com/svtrostov/oclexplorer">https://github.com/svtrostov/oclexplorer</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://proglib.io/p/new-bitcoin">https://proglib.io/p/new-bitcoin</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D0%BB%D0%BB%D0%B8%D0%B7%D0%B8%D1%8F_%D1%85%D0%B5%D1%88-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%B8">https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D0%BB%D0%BB%D0%B8%D0%B7%D0%B8%D1%8F_%D1%85%D0%B5%D1%88-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%B8</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/319868/">https://habr.com/ru/articles/319868/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://science-engineering.ru/ru/article/view?id=1247">https://science-engineering.ru/ru/article/view?id=1247</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python">https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/svtrostov/oclexplorer/issues/6">https://github.com/svtrostov/oclexplorer/issues/6</a></li>
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
