# How to Check Bitcoin Address Balance Using Bloom Filter Algorithm

<!-- wp:image {"id":2533,"sizeSlug":"large","linkDestination":"none"} -->
<figure class="wp-block-image size-large"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-8-1024x519.png" alt="" class="wp-image-2533"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4 class="wp-block-heading"><a href="https://polynonce.ru/author/polynonce/"></a></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To create a Python script that checks the balance of Bitcoin addresses using the Bloom filter algorithm, we need to follow a few steps. First, we will create a Bloom filter to store the addresses, and then we will check the balance of each address using the Blockchain.com API.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 1: Installing the required libraries</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To work, we will need libraries&nbsp;<code>pybloom_live</code>for implementing the Bloom filter and&nbsp;<code>requests</code>for API requests.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">bash<code>pip install pybloom_live requests
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 2: Create a script</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Below is an example script that creates a Bloom filter and checks the balance of Bitcoin addresses:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import requests
from pybloom_live import BloomFilter

<em># Параметры для Bloom filter</em>
size = 1000000  <em># Размер фильтра</em>
error_rate = 0.001  <em># Вероятность ложного срабатывания</em>

<em># Создание Bloom filter</em>
bf = BloomFilter(size, error_rate)

<em># Список Bitcoin адресов</em>
addresses = [
    '12VeK1eRgPHRUikNLXq3Nuz99gS2S46QMD',
    <em># Добавьте больше адресов сюда</em>
]

<em># Добавление адресов в Bloom filter</em>
for address in addresses:
    bf.add(address)

<em># Функция для проверки баланса адреса</em>
def check_balance(address):
    url = f'https://blockchain.info/rawaddr/{address}'
    try:
        response = requests.get(url)
        data = response.json()
        balance = data['final_balance'] / 1e8  <em># Переводим в BTC</em>
        print(f'Адрес: {address}, Баланс: {balance} BTC')
    except Exception as e:
        print(f'Ошибка при запросе баланса для {address}: {e}')

<em># Проверка баланса для каждого адреса в Bloom filter</em>
for address in bf:
    if address in addresses:  <em># Проверяем, чтобы адрес был в исходном списке</em>
        check_balance(address)

<em># Если вы хотите проверить баланс для новых адресов, не добавленных в Bloom filter</em>
new_address = '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'
if new_address not in bf:
    bf.add(new_address)
    check_balance(new_address)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Notes</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bloom Filter</strong> : Used to quickly determine if an element is present in a set. However, it can produce false positives, so it is important to double-check the results.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Blockchain.com API</strong> : Used to get Bitcoin address balances. Please note that there are limits on the number of requests per minute.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Adding addresses</strong> : In the script you need to manually add Bitcoin addresses to the list <code>addresses</code>.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>This script allows you to quickly filter addresses that have already been verified and then check their balance. However, to fully use the Bloom filter in real applications, you need to carefully adjust its size and error probability depending on your specific requirements.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading"><br>How to Use Bloom Filter to Check Bitcoin Address Balance</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To check the balance of Bitcoin addresses using Bloom filter, you can follow these steps:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 1: Implementing Bloom Filter</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>First we need to implement or use a ready-made implementation of Bloom filter. We can use a library&nbsp;<code>pybloom_live</code>for Python.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>from pybloom_live import BloomFilter

<em># Параметры для Bloom filter</em>
size = 1000000  <em># Размер фильтра</em>
error_rate = 0.001  <em># Вероятность ложного срабатывания</em>

<em># Создание Bloom filter</em>
bf = BloomFilter(size, error_rate)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 2: Adding Addresses to Bloom Filter</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Add Bitcoin addresses to the Bloom filter. This will allow you to quickly check if the address is in the set.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code><em># Список Bitcoin адресов</em>
addresses = [
    '12VeK1eRgPHRUikNLXq3Nuz99gS2S46QMD',
    <em># Добавьте больше адресов сюда</em>
]

<em># Добавление адресов в Bloom filter</em>
for address in addresses:
    bf.add(address)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 3: Checking the balance of addresses</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Use the Blockchain.com API to check the balance of each address. If the address is in the Bloom filter, check its balance.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import requests

<em># Функция для проверки баланса адреса</em>
def check_balance(address):
    url = f'https://blockchain.info/rawaddr/{address}'
    try:
        response = requests.get(url)
        data = response.json()
        balance = data['final_balance'] / 1e8  <em># Переводим в BTC</em>
        print(f'Адрес: {address}, Баланс: {balance} BTC')
    except Exception as e:
        print(f'Ошибка при запросе баланса для {address}: {e}')

<em># Проверка баланса для каждого адреса в Bloom filter</em>
for address in addresses:
    if address in bf:
        check_balance(address)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Notes</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bloom Filter</strong> : Used to quickly determine if an element is present in a set. However, it can produce false positives, so it is important to double-check the results.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Blockchain.com API</strong> : Used to get Bitcoin address balances. Please note that there are limits on the number of requests per minute.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Adding addresses</strong> : In the script you need to manually add Bitcoin addresses to the list <code>addresses</code>.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>This approach allows you to quickly filter addresses that have already been verified and then check their balance. However, to fully use the Bloom filter in real applications, you need to carefully tune its size and error probability depending on specific requirements.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Full script</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>from pybloom_live import BloomFilter
import requests

<em># Параметры для Bloom filter</em>
size = 1000000  <em># Размер фильтра</em>
error_rate = 0.001  <em># Вероятность ложного срабатывания</em>

<em># Создание Bloom filter</em>
bf = BloomFilter(size, error_rate)

<em># Список Bitcoin адресов</em>
addresses = [
    '12VeK1eRgPHRUikNLXq3Nuz99gS2S46QMD',
    <em># Добавьте больше адресов сюда</em>
]

<em># Добавление адресов в Bloom filter</em>
for address in addresses:
    bf.add(address)

<em># Функция для проверки баланса адреса</em>
def check_balance(address):
    url = f'https://blockchain.info/rawaddr/{address}'
    try:
        response = requests.get(url)
        data = response.json()
        balance = data['final_balance'] / 1e8  <em># Переводим в BTC</em>
        print(f'Адрес: {address}, Баланс: {balance} BTC')
    except Exception as e:
        print(f'Ошибка при запросе баланса для {address}: {e}')

<em># Проверка баланса для каждого адреса в Bloom filter</em>
for address in addresses:
    if address in bf:
        check_balance(address)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This script allows you to quickly filter addresses and check their balance, using Bloom filter to optimize the process.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What Bloom filter parameters should be taken into account when working with Bitcoin addresses</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>When working with Bitcoin addresses and using Bloom filter, the following key parameters must be taken into account:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Bit array size (m)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : This is the number of bits allocated to store information about the presence of elements.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Impact</strong> : The larger the value <code>m</code>, the lower the probability of false positives, but more memory will be required.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>The formula for calculation</strong> is: m=−n⋅ln⁡(p)(ln⁡(2))2m = -\frac{n \cdot \ln(p)}{(\ln(2))^2}m=−(ln(2))2n⋅ln(p), where nnn is the expected number of elements and ppp is the desired probability of false positives <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">5 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Number of hash functions (k)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : The number of hash functions used to calculate indices into the bitmap.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Impact</strong> : Increasing <code>k</code>reduces the number of false positives, but slows down the checking process.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>The formula for calculation</strong> : k=mn⋅ln⁡(2)k = \frac{m}{n} \cdot \ln(2)k=nm⋅ln(2), where mmm is the size of the bit array, and nnn is the number of elements <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">5 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Probability of false positives (p)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : The probability that the filter will report the presence of an element that is not actually present.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Impact</strong> : Decreasing <code>p</code>requires increasing <code>m</code>and <code>k</code>, which may impact performance and memory consumption <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">5 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Number of elements (n)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : The expected number of Bitcoin addresses that will be stored in the filter.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Impact</strong> : Affects the size of the bit array and the number of hash functions required to achieve the desired accuracy <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">of 5 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Hash functions</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Used to calculate indices in a bit array.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Impact</strong> : The choice of hash functions affects the performance and uniformity of the distribution of elements in the <a href="https://habr.com/ru/companies/otus/articles/843714/" target="_blank" rel="noreferrer noopener">5 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">8</a> filter .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>When working with Bitcoin addresses, it is important to find a balance between memory, speed, and filter accuracy to ensure effective filtering and minimize false positives.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What errors can occur when using Bloom filter to check the balance of Bitcoin addresses</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>When using Bloom filter to check Bitcoin address balances, the following errors may occur:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>False positives (false positives)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : A Bloom filter may report that an address is present in a set even if it is not. This is due to hash function collisions, where different inputs produce the same hash <a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Misidentification of addresses as having a balance when in fact they do not.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Insufficient memory</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : If the bitmap size is too small for the number of addresses stored, this may result in an increased probability of false positives <a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Reduced filter accuracy, which may lead to incorrect results.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Incorrect choice of hash functions</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : If hash functions do not distribute elements uniformly across the bitmap, this can increase the likelihood of collisions <a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Increased number of false positives.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Incorrect parameter settings</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Incorrect choice of the number of hash functions ( <code>k</code>) or the size of the bit array ( <code>m</code>) can lead to suboptimal performance of the filter <a href="https://habr.com/ru/articles/491132/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Reduced filter performance or accuracy.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Scalability issues</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : When working with a large number of addresses, the Bloom filter can require significant memory resources, which can be problematic on devices with limited resources <a href="https://habr.com/ru/articles/491132/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Reduced system performance due to memory consumption.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">6.&nbsp;<strong>Incorrect API error handling</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : If an API error (such as 500 or 503) occurs when requesting a balance, this may result in incorrect results or the program freezing <a href="https://github.com/Blockchair/Blockchair.Support/blob/master/API_DOCUMENTATION_RU.md" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Program execution may be interrupted or incorrect data may be received.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>To minimize these errors, it is important to carefully configure the Bloom filter parameters and implement correct error handling when making API requests.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What collisions can occur when using Bloom filter for Bitcoin addresses</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>When using Bloom filter for Bitcoin addresses the following collisions may occur:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>False positives (false positives)</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Bloom filter can report that a Bitcoin address is present in the set even if it is not. This happens due to hash function collisions, when different addresses give the same hash <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Misidentification of addresses as having a balance when in fact they do not.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Hash function collisions</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description : Two different Bitcoin addresses can have the same hash, which results in the same indices in the </strong><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">4</a> Bloom filter bitmap .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Increased likelihood of false positives, which may lead to incorrect results when checking balance.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Insufficient bit array size</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : If the bitmap size is too small for the number of addresses stored, this may result in an increased chance of collisions <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Reduced filter accuracy, which may lead to incorrect results.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Incorrect choice of hash functions</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : If hash functions do not distribute elements evenly across the bitmap, this can increase the likelihood of collisions <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Increased number of false positives.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>To reduce the likelihood of collisions, you can use the following strategies:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Increasing the size of the bit array</strong> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Increasing the number of hash functions</strong> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Selecting quality hash functions</strong> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Using specialized algorithms</strong> such as Counting Bloom Filter <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">What strategies can be used to reduce collisions in Bloom filter</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To reduce collisions in Bloom filter, the following strategies can be used:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1.&nbsp;<strong>Increasing the size of the bit array</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Increasing the bitmap size ( <code>m</code>) reduces the chance of collisions because the number of bits available for storing hash codes increases.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Requires more memory, but reduces the likelihood of false positives <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2.&nbsp;<strong>Increasing the number of hash functions</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Using more hash functions ( <code>k</code>) increases the number of bits that are marked when an element is added, reducing the likelihood of false positives.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Consequences</strong> : Increases the number of memory accesses, but improves the accuracy of the <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://habr.com/ru/articles/491132/" target="_blank" rel="noreferrer noopener">3</a> filter .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3.&nbsp;<strong>Selecting quality hash functions</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Using hash functions that evenly distribute elements across a bitmap reduces the likelihood of collisions.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Effect</strong> : Improves filter accuracy and reduces false positives <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1 </a><a href="https://dzen.ru/a/XZR3S9W7wwCsGo3R" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">4.&nbsp;<strong>Using specialized algorithms</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Using algorithms such as the Counting Bloom Filter, it is possible to track the number of times each element is encountered, which helps reduce the percentage of false positives.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Effect</strong> : Improves filter accuracy and allows more accurate checking of the presence of elements <a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">5.&nbsp;<strong>Preliminary check</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Use Bloom filter as a pre-filter to exclude elements that are definitely not in a set, and then confirm the results with more accurate methods.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Impact</strong> : Reduces the number of queries to the main database and speeds up the verification process <a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These strategies allow you to optimize the Bloom filter and minimize the number of collisions, which is especially important when working with large data sets.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike">https://gitverse.ru/blog/articles/data/255-chto-takoe-filtr-bluma-i-kak-on-rabotaet-na-praktike</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/491132/">https://habr.com/ru/articles/491132/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/XZR3S9W7wwCsGo3R">https://dzen.ru/a/XZR3S9W7wwCsGo3R</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://evmservice.ru/blog/filtr-bluma/">https://evmservice.ru/blog/filtr-bluma/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://docs.unity3d.com/ru/530/Manual/script-BloomOptimized.html">https://docs.unity3d.com/ru/530/Manual/script-BloomOptimized.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
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
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://datafinder.ru/files/downloads/01/algoritmy_i_struktury_dlja_massivnyh_naborov_dannyh_2023_medzhedovich.pdf">https://datafinder.ru/files/downloads/01/algoritmy_i_struktury_dlja_massivnyh_naborov_dannyh_2023_medzhedovich.pdf</a></li>
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
<li><a href="https://github.com/Blockchair/Blockchair.Support/blob/master/API_DOCUMENTATION_RU.md">https://github.com/Blockchair/Blockchair.Support/blob/master/API_DOCUMENTATION_RU.md</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/491132/">https://habr.com/ru/articles/491132/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pikabu.ru/story/istoricheskaya_spravka_kasaemo_vzlomov_bitkoin_koshelkov_9596225">https://pikabu.ru/story/istoricheskaya_spravka_kasaemo_vzlomov_bitkoin_koshelkov_9596225</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://datafinder.ru/files/downloads/01/algoritmy_i_struktury_dlja_massivnyh_naborov_dannyh_2023_medzhedovich.pdf">https://datafinder.ru/files/downloads/01/algoritmy_i_struktury_dlja_massivnyh_naborov_dannyh_2023_medzhedovich.pdf</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://nuancesprog.ru/p/21154/">https://nuancesprog.ru/p/21154/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ibmm.ru/news/chto-novogo-v-bitcoin-core-0-21-0/">https://ibmm.ru/news/chto-novogo-v-bitcoin-core-0-21-0/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/843714/">https://habr.com/ru/companies/otus/articles/843714/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forklog.com/exclusive/bitkoin-koshelki-sravnili-po-48-kriteriyam-najdite-svoj">https://forklog.com/exclusive/bitkoin-koshelki-sravnili-po-48-kriteriyam-najdite-svoj</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/brichard19/BitCrack/issues/313">https://github.com/brichard19/BitCrack/issues/313</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://crypto.oni.su/54-python-perebor-mnemonicheskih-fraz-dlja-btcethtrx.html">https://crypto.oni.su/54-python-perebor-mnemonicheskih-fraz-dlja-btcethtrx.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/brichard19/BitCrack/issues/313">https://github.com/brichard19/BitCrack/issues/313</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.bitget.com/ru/glossary/bloom-filter">https://www.bitget.com/ru/glossary/bloom-filter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forklog.com/exclusive/bitkoin-koshelki-sravnili-po-48-kriteriyam-najdite-svoj">https://forklog.com/exclusive/bitkoin-koshelki-sravnili-po-48-kriteriyam-najdite-svoj</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0">https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%91%D0%BB%D1%83%D0%BC%D0%B0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/YBkKO40wyxeAFLPO">https://dzen.ru/a/YBkKO40wyxeAFLPO</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://cryptodeep.ru/check-bitcoin-address-balance/">https://cryptodeep.ru/check-bitcoin-address-balance/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://rutube.ru/video/4559c67d6deb70128512cbf232bb8d4e/">https://rutube.ru/video/4559c67d6deb70128512cbf232bb8d4e/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.youtube.com/watch?v=LrVLVyaeMRA">https://www.youtube.com/watch?v=LrVLVyaeMRA</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/">https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.linux.org.ru/forum/development/14449143">https://www.linux.org.ru/forum/development/14449143</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/674812/">https://habr.com/ru/articles/674812/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/OxideDevX/btcbruter_script">https://github.com/OxideDevX/btcbruter_script</a></li>
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
