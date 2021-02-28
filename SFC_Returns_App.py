import investpy
import datetime
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st
import os
from bokeh.models import Div
pd.options.display.max_columns = 500
from PIL import Image


password = st.sidebar.text_input("Password:", value="", type="password")

# select our text input field and make it into a password input
js = "el = document.querySelectorAll('.sidebar-content input')[0]; el.type = 'password';"

# passing js code to the onerror handler of an img tag with no src
# triggers an error and allows automatically running our code
html = f'<img src onerror="{js}">'

# in contrast to st.write, this seems to allow passing javascript
div = Div(text=html)
st.bokeh_chart(div)

if password != str('Voldemort88'):
    st.error("the password you entered is incorrect")
elif password == str('Voldemort88'):
    st.sidebar.title('HPE Portfolio')

    app_mode = st.sidebar.selectbox("Choose the app mode",
            ["SFC Energy", "Portfolio Company 2","Portfolio Company 3"])
    if app_mode == "SFC Energy":

        st.write("""
        # SFC Energy Demo
        Demo Data Web app,  ***Suggesties voor Tom?***
        """)

        # dd/mm/YY
        today_datetime = datetime.date.today()
        st.subheader('Start Date for Data Analysis')
        input_date=st.date_input('', value=datetime.date(2021, 1, 1),max_value=today_datetime)
        start_date=input_date.strftime("%d/%m/%Y")
        today = date.today().strftime("%d/%m/%Y")

        stock_list=['FCEL','BLDP','PLUG','F3CG']
        country_list=['United States','United States','United States','germany']


        #selected_stock_list=st.multiselect('Select Stocks', stock_list)

        df_sfc = investpy.get_stock_historical_data(stock='F3CG',
                                                country='germany',
                                                from_date=start_date,
                                                to_date=today)
        df_sfc=pd.DataFrame(df_sfc.to_records())
        df_sfc['Change']=((df_sfc['Close']-df_sfc['Close'].iloc[0])/df_sfc['Close'].iloc[0])*100
        df_sfc=df_sfc.add_suffix('_SFC')

        df_sp = investpy.get_etf_historical_data(etf='SPDR S&P 500',country='United States', from_date=start_date, to_date=today)

        df_sp=pd.DataFrame(df_sp.to_records())
        df_sp['Change']=((df_sp['Close']-df_sp['Close'].iloc[0])/df_sp['Close'].iloc[0])*100
        df_sp=df_sp.add_suffix('_S&P500')

        def df_stock1(stock, country):
            df_value=investpy.get_stock_historical_data(stock=stock,
                                                    country=country,
                                                    from_date=start_date,
                                                    to_date=today)
            df_value=pd.DataFrame(df_value.to_records())
            df_value['Change']=((df_value['Close']-df_value['Close'].iloc[0])/df_value['Close'].iloc[0])*100
            return df_value
        #df_stock1('FCEL')

        def df_stock2(stock,country):
            df_value=df_stock1(stock,country)
            new_names = [(i,i+'_'+str(stock)) for i in df_value.iloc[:, 1:].columns.values]
            df_value.rename(columns = dict(new_names), inplace=True)
            return df_value
        #df_stock2('FCEL')

        stock_list=['FCEL','BLDP','PLUG']
        country_list=['United States','United States','United States']

        df = pd.DataFrame(columns=['Date'])

        for stock,country in zip(stock_list,country_list):
            df_value=df_stock2(stock,country)
            df=df.merge(df_value, on='Date', how='outer')   

        df_all2=df.merge(df_sfc,left_on='Date', right_on='Date_SFC', how='left')
        df_all3=df_all2.merge(df_sp, left_on='Date', right_on='Date_S&P500', how='left')
        df_all3['Year']=df_all3['Date'].dt.year
        df_all3['Month']=df_all3['Date'].dt.month

        change_cols = [col for col in df_all3.columns if 'Change' in col]

        df_change = df_all3[df_all3.columns.intersection(change_cols)]

        df_recent_change=pd.DataFrame(df_all3.iloc[:,0]).merge(df_change, left_index=True, right_index=True, how='left')

        fig = px.line(df_recent_change, x="Date", y=change_cols)
        fig.update_layout(
            title="% Cum Gain/Loss Since Start Date ",
            xaxis_title="",
            yaxis_title="% Cum Gain/Loss",
            legend_title="Stock/ETF",
            font=dict(
                #family="Times New Roman, monospace",
                size=15,
                color="black"
            )
        )

        close_cols = [col for col in df_all3.columns if 'Close' in col]
        close_cols2=[x for x in close_cols if "S&P" not in x]
        df_close=df_all3[df_all3.columns.intersection(close_cols2)]
        df_close_use=pd.DataFrame(df_all3.iloc[:,0]).merge(df_close, left_index=True, right_index=True, how='left')
        fig2 = px.line(df_close_use, x="Date", y=close_cols2)
        fig2.update_layout(
            title="Historic Daily Closing Stock Price ",
            xaxis_title="",
            yaxis_title="Closing Stock Price ($ or Euro)",
            legend_title="Stock",
            font=dict(
                #family="Times New Roman, monospace",
                size=15,
                color="black"
            )
        )


        # Earnings Statements
        def df_rev1(stock,country):
            df_rev=investpy.get_stock_financial_summary(stock=stock, country=country, summary_type='income_statement', period='quarterly')
            df_rev=pd.DataFrame(df_rev.to_records())
            new_names = [(i,i+'_'+str(stock)) for i in df_rev.iloc[:, 1:].columns.values]
            df_rev.rename(columns = dict(new_names), inplace=True)
            return df_rev
        #df_rev1('FCEL')

        df2 = pd.DataFrame(columns=['Date'])

        #Earnings
        earnings_stock_list=['FCEL','BLDP','PLUG','F3CG']
        earnings_country_list=['United States','United States','United States','germany']

        for stock,country in zip(earnings_stock_list,earnings_country_list):
            df_value2=df_rev1(stock,country)
            df2=df2.merge(df_value2, on='Date', how='outer')  
        df2=df2.sort_values('Date')
        df2.columns = df2.columns.str.replace("F3CG", "SFC")

        revenue_cols = [col for col in df2.columns if 'Revenue' in col]
        df_revenue=df2[df2.columns.intersection(revenue_cols)]
        df_revenue_use=pd.DataFrame(df2.iloc[:,0]).merge(df_revenue, left_index=True, right_index=True, how='left')
        fig3 = px.scatter(df_revenue_use, x="Date", y=revenue_cols)
        fig3.update_layout(
            title="Last 4 Earnings Revenue ",
            xaxis_title="",
            yaxis_title="Quarterly Revnue ($ or Euro)",
            legend_title="Stock",
            font=dict(
                #family="Times New Roman, monospace",
                size=15,
                color="black"
            )
        )

        st.write("""

        """)

        st.plotly_chart(fig)

        st.plotly_chart(fig2)

        st.subheader('Earnings Statements')

        st.plotly_chart(fig3)

        earnings_stock_names=['Fuel Cell (FCEL)', 'Ballard Power (BLDP)', 'Plug Power (PLUG)', 'SFC Energy (F3CG)']
        earnings_stock_list=['FCEL','BLDP','PLUG','F3CG']
        earnings_country_list=['United States','United States','United States','germany']
        df_zip=pd.DataFrame({'company':earnings_stock_names, 'symbol': earnings_stock_list,'country': earnings_country_list})
        

        stock_pick=st.selectbox(label="Select Company's earnings", options=earnings_stock_names)

        def table_earnings(stock,country):
            df_rev=investpy.get_stock_financial_summary(stock=stock, country=country, summary_type='income_statement', period='quarterly')
            df_rev=pd.DataFrame(df_rev.to_records())
            return df_rev

        df_earnings=table_earnings(df_zip[df_zip['company']==stock_pick]['symbol'].iloc[0],df_zip[df_zip['company']==stock_pick]['country'].iloc[0])
        df_earnings



    elif app_mode == "Portfolio Company 2":
        st.write("""
        # Potential More HPE Portfolio Companies
        Waiting on Feedback
        """)

    elif app_mode == "Portfolio Company 3":
        st.write("""
        # Potential More HPE Portfolio Companies
        Waiting on Feedback
        """)
