import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title='Startup Analysis',
    layout='wide'
)

# Load and clean dataset
df = pd.read_csv('startup_clean_final.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['investors'] = df['investors'].fillna('Undisclosed')

# Standardize common duplicate city and sector labels
df['city'] = df['city'].replace({
    'Bangalore': 'Bengaluru',
    'New Delhi': 'Delhi',
    'Gurgaon': 'Gurugram'
})
df['vertical'] = df['vertical'].replace({
    'ECommerce': 'E-commerce'
})

st.sidebar.title("Startup Funding Analysis")
option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'Startup', 'Investor'])


def load_investor_details(investor):
    st.title(investor)

    # Filter rows where this investor participated (na=False prevents NaN crashes)
    inv_mask = df['investors'].str.contains(investor, na=False, regex=False)

    # Load recent 5 investments
    last5_df = df[inv_mask].sort_values(by='date', ascending=False).head()[
        ['date', 'startup', 'vertical', 'city', 'round', 'amount']
    ]
    st.subheader('Most Recent Investments')
    st.dataframe(last5_df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        # Biggest investments
        big_series = (
            df[inv_mask]
            .groupby('startup')['amount']
            .sum()
            .sort_values(ascending=False)
            .head()
        )
        st.subheader("Biggest Investments")
        fig, ax = plt.subplots()
        ax.bar(big_series.index, big_series.values)
        ax.set_ylabel("Amount (Cr)")
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

    with col2:
        # Fixed: uses variable `investor` instead of string `'investors'`
        vertical_series = df[inv_mask].groupby('vertical')['amount'].sum()
        st.subheader('Sectors Invested In')
        fig1, ax1 = plt.subplots()
        ax1.pie(vertical_series.values, labels=vertical_series.index, autopct='%1.1f%%')
        st.pyplot(fig1)

    col3, col4 = st.columns(2)
    with col3:
        stage = (
            df[inv_mask]
            .groupby('round')['amount']
            .sum()
            .sort_values()
        )
        st.subheader("Stage Invested In")
        fig2, ax2 = plt.subplots()
        ax2.barh(stage.index, stage.values)
        ax2.set_xlabel("Total Investment (Cr)")
        ax2.set_ylabel("Funding Stage")
        st.pyplot(fig2)

    with col4:
        city = (
            df[inv_mask]
            .groupby('city')['amount']
            .sum()
            .sort_values()
        )
        st.subheader("City Invested In")
        fig3, ax3 = plt.subplots()
        ax3.barh(city.index, city.values)
        ax3.set_xlabel("Total Investment (Cr)")
        ax3.set_ylabel("City")
        st.pyplot(fig3)

    year_series = df[inv_mask].groupby('year')['amount'].sum()
    st.subheader('YoY Investment')
    fig4, ax4 = plt.subplots()
    ax4.plot(year_series.index, year_series.values, marker='o')
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Total Investment (Cr)")
    st.pyplot(fig4)


def load_overall_analysis():
    st.title("Overall Analysis")
    total = round(df['amount'].sum())

    # Maximum amount infused in a startup
    max_funding = round(df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0], 2)
    avg_funding = round(df.groupby('startup')['amount'].sum().mean(), 2)
    num_startups = df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total', f"{total} Cr")
    with col2:
        st.metric('Maximum Funding', f"{max_funding} Cr")
    with col3:
        st.metric("Average Funding", f"{avg_funding} Cr")
    with col4:
        st.metric('Funded Startups', str(num_startups))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        'Sector Analysis',
        'Type of Funding',
        'City-Wise Funding',
        'Top Startups',
        'Top Investors',
        'MoM Analysis',
        'Funding HeatMap'
    ])

    with tab1:
        st.header('Sector Analysis')
        sector_type = st.selectbox(
            "Select Analysis Type",
            ['Total Funding(Sum)', 'Deal Count'],
            key='sector_select'
        )
        if sector_type == 'Total Funding(Sum)':
            sector_series = df.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(10)
        else:
            sector_series = df.groupby('vertical')['amount'].count().sort_values(ascending=False).head(10)
        fig5, ax5 = plt.subplots()
        ax5.pie(sector_series, labels=sector_series.index, autopct='%0.1f%%')
        st.pyplot(fig5)

    with tab2:
        st.header('Type of Funding')
        round_df = (
            df.groupby('round')['amount']
            .agg(
                Deals='count',
                Total_Funding_Cr='sum',
                Avg_Ticket_Cr='mean',
                Median_Deal_Cr='median',
                Max_Deal_Cr='max'
            )
            .sort_values(by='Deals', ascending=False)
            .head(10)
            .round(2)
        )
        st.dataframe(round_df)

    with tab3:
        st.header('City-Wise Funding')
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Top Cities By Funding (Cr)")
            citysum = df.groupby('city')['amount'].sum().sort_values(ascending=False).head(10).round(2)
            st.dataframe(citysum)
        with col6:
            st.subheader("Top Cities By Deals")
            citycount = df.groupby('city')['amount'].count().sort_values(ascending=False).head(10)
            st.dataframe(citycount)

    with tab4:
        st.header('Top Startups')
        stats = (
            df.groupby('startup')['amount']
            .agg(Total_Funding='sum', Rounds='count')
            .sort_values(by='Total_Funding', ascending=False)
            .head(10)
            .round(2)
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Funding vs. Rounds')
            fig8, ax8 = plt.subplots()
            ax8.scatter(stats['Rounds'], stats['Total_Funding'], s=100, color='#1f77b4')
            for startup, row in stats.iterrows():
                ax8.annotate(
                    startup,
                    (row['Rounds'], row['Total_Funding']),
                    fontsize=8,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
            ax8.set_xlabel('Number of Rounds (Count)')
            ax8.set_ylabel('Total Funding (Cr)')
            st.pyplot(fig8)

        with col2:
            st.subheader('Top Startups Breakdown')
            st.dataframe(stats)

    with tab5:
        st.header('Top Investors')

        # Split comma-separated investors into individual rows
        inv_df = df.dropna(subset=['investors']).copy()
        inv_df['investors'] = inv_df['investors'].str.split(',')
        inv_df = inv_df.explode('investors')
        inv_df['investors'] = inv_df['investors'].str.strip()

        # Remove blank or undisclosed entries
        inv_df = inv_df[
            ~inv_df['investors'].isin(['', 'Undisclosed Investors', 'Undisclosed', 'undisclosed investors'])
        ]

        top_inv = (
            inv_df.groupby('investors')['amount']
            .agg(Deals='count', Total_Funding_Cr='sum')
            .sort_values(by='Deals', ascending=False)
            .head(10)
            .round(2)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Most Active Investors (by Deals)')
            fig9, ax9 = plt.subplots()
            ax9.barh(top_inv.index[::-1], top_inv['Deals'][::-1])
            ax9.set_xlabel('Number of Deals')
            st.pyplot(fig9)

        with col2:
            st.subheader('Investor Breakdown')
            st.dataframe(top_inv)

    with tab6:
        st.header('Month on Month Analysis Graph')
        selected_option = st.selectbox('Select Type', ['Total', 'Count'], key='mom_select')
        if selected_option == "Total":
            temp_df = df.groupby(['year', 'month'])['amount'].sum().reset_index()
        else:
            temp_df = df.groupby(['year', 'month'])['amount'].count().reset_index()

        temp_df['x-axis'] = temp_df['month'].astype('str') + '-' + temp_df['year'].astype('str')
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(temp_df['x-axis'], temp_df['amount'])
        ax3.tick_params(axis='x', rotation=90)
        st.pyplot(fig3)

    with tab7:
        st.header('Funding Heatmap (Year vs. Month)')
        heatmap_type = st.selectbox(
            'Select Heatmap Metric',
            ['Total Funding (Sum)', 'Deal Count'],
            key='heatmap_select'
        )

        if heatmap_type == 'Total Funding (Sum)':
            pivot_df = df.pivot_table(
                index='year',
                columns='month',
                values='amount',
                aggfunc='sum',
                fill_value=0
            ).round(0)
            fmt_str = '.0f'
        else:
            pivot_df = df.pivot_table(
                index='year',
                columns='month',
                values='amount',
                aggfunc='count',
                fill_value=0
            )
            fmt_str = 'd'

        fig10, ax10 = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_df, annot=True, fmt=fmt_str, cmap='YlGnBu', ax=ax10)
        ax10.set_xlabel('Month')
        ax10.set_ylabel('Year')
        st.pyplot(fig10)


def load_startup_details(startup):
    st.title(startup)

    # Filter dataframe for the selected startup
    startup_df = df[df['startup'] == startup]

    # 1. Basic Info Cards (Industry, Subindustry, Location, Total Raised)
    industry = startup_df['vertical'].dropna().values[0] if not startup_df['vertical'].dropna().empty else 'N/A'
    subindustry = startup_df['subvertical'].dropna().values[0] if not startup_df['subvertical'].dropna().empty else 'N/A'
    location = startup_df['city'].dropna().values[0] if not startup_df['city'].dropna().empty else 'N/A'
    total_raised = round(startup_df['amount'].sum(), 2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Industry', industry)
    with col2:
        st.metric('Subindustry', subindustry)
    with col3:
        st.metric('Location', location)
    with col4:
        st.metric('Total Raised', f"{total_raised} Cr")

    # 2. Funding Rounds (Stage, Investors, Date, Amount)
    st.subheader('Funding Rounds')
    rounds_df = (
        startup_df[['date', 'round', 'investors', 'amount']]
        .sort_values(by='date', ascending=False)
        .rename(columns={
            'date': 'Date',
            'round': 'Stage',
            'investors': 'Investors',
            'amount': 'Amount (Cr)'
        })
    )
    st.dataframe(rounds_df, hide_index=True)

    # 3. Similar Companies (same subvertical first, fallback to vertical)
    st.subheader('Similar Companies')
    if subindustry != 'N/A':
        similar_df = df[(df['subvertical'] == subindustry) & (df['startup'] != startup)]
    else:
        similar_df = df[(df['vertical'] == industry) & (df['startup'] != startup)]

    if not similar_df.empty:
        similar_companies = (
            similar_df.groupby('startup')['amount']
            .agg(Total_Funding_Cr='sum', Rounds='count')
            .sort_values(by='Total_Funding_Cr', ascending=False)
            .head(5)
            .round(2)
        )
        st.dataframe(similar_companies)
    else:
        st.info('No similar startups found in this subindustry.')


def investor_clustering():
    st.title("Similar Investors")

    # Create investor-level data
    investor_data = df.copy()
    investor_data['investors'] = investor_data['investors'].str.split(',')
    investor_data = investor_data.explode('investors')
    investor_data['investors'] = investor_data['investors'].str.strip()

    # Filter out blank and undisclosed rows before clustering
    investor_data = investor_data[
        ~investor_data['investors'].isin(['', 'Undisclosed', 'Undisclosed Investors', 'undisclosed investors'])
    ]

    # Create features for each investor
    investor_features = (
        investor_data.groupby('investors')
        .agg(
            total_investment=('amount', 'sum'),
            startups_invested=('startup', 'nunique'),
            sectors=('vertical', 'nunique'),
            cities=('city', 'nunique'),
            rounds=('round', 'nunique')
        )
        .reset_index()
    )

    # Features for clustering
    X = investor_features[
        ['total_investment', 'startups_invested', 'sectors', 'cities', 'rounds']
    ]

    # Scale and run KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    investor_features['cluster'] = kmeans.fit_predict(X_scaled)

    # Select investor
    selected = st.selectbox(
        "Select Investor",
        sorted(investor_features['investors'])
    )

    cluster_number = investor_features[
        investor_features['investors'] == selected
    ]['cluster'].values[0]

    similar = investor_features[
        investor_features['cluster'] == cluster_number
    ].sort_values(by='total_investment', ascending=False)

    st.subheader("Investors with Similar Investment Behaviour")
    st.dataframe(similar, hide_index=True)


# Sidebar navigation routing
if option == 'Overall Analysis':
    load_overall_analysis()

elif option == 'Startup':
    selected_startup = st.sidebar.selectbox(
        'Select Startup',
        sorted(df['startup'].dropna().unique().tolist())
    )
    btn1 = st.sidebar.button('Find Startup Details')
    if btn1:
        load_startup_details(selected_startup)

else:
    investor_option = st.sidebar.selectbox(
        'Select Option',
        ['Investor Details', 'Similar Investors']
    )

    if investor_option == 'Investor Details':
        investor_list = sorted(
            [
                inv for inv in df['investors'].str.split(',').explode().str.strip().dropna().unique()
                if inv not in ['', 'Undisclosed', 'Undisclosed Investors', 'undisclosed investors']
            ]
        )
        selected_investor = st.sidebar.selectbox('Select Investor', investor_list)
        btn2 = st.sidebar.button('Find Investor Details')
        if btn2:
            load_investor_details(selected_investor)
    else:
        investor_clustering()